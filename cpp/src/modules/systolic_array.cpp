#include "npu_sim/modules/systolic_array.hpp"

#include <algorithm>
#include <cmath>
#include <deque>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace npu_sim {

namespace {

void validate_numeric_format(const NumericFormatConfig& format) {
  if (format.bits == 0 || format.bits > 63) {
    throw std::invalid_argument("numeric format bits must be in [1, 63]");
  }

  if (format.kind == NumericFormatKind::Float) {
    if (format.exponent_bits == 0 || format.mantissa_bits == 0) {
      throw std::invalid_argument(
          "floating format exponent_bits and mantissa_bits must be positive");
    }
    if (format.exponent_bits + format.mantissa_bits >= format.bits) {
      throw std::invalid_argument(
          "floating format exponent_bits + mantissa_bits must leave room for "
          "sign or encoding bits");
    }
  }
}

PEValue zero_value_for_format(const NumericFormatConfig& format) {
  if (format.kind == NumericFormatKind::Float) {
    return PEValue::floating(format, 0.0);
  }
  return PEValue::integer(format, 0);
}

double value_as_double(const PEValue& value) {
  return value.is_float() ? value.floating_value()
                          : static_cast<double>(value.integer_value());
}

const char* numeric_format_kind_name(NumericFormatKind kind) {
  switch (kind) {
    case NumericFormatKind::SignedInt:
      return "signed_int";
    case NumericFormatKind::UnsignedInt:
      return "unsigned_int";
    case NumericFormatKind::Float:
      return "float";
  }
  throw std::invalid_argument("unknown numeric format kind");
}

struct StreamItem {
  StreamItem(Cycle cycle, std::int64_t row, std::int64_t col, std::int64_t k,
             PEValue value)
      : cycle(cycle),
        row(row),
        col(col),
        k(k),
        value(std::move(value)) {}

  Cycle cycle;
  std::int64_t row;
  std::int64_t col;
  std::int64_t k;
  PEValue value;
};

using StreamFifo = std::deque<StreamItem>;

}  // namespace

SystolicArray::SystolicArray(sc_core::sc_module_name module_name,
                             SystolicArrayConfig config)
    : sc_core::sc_module(module_name), config_(std::move(config)) {
  validate_config();

  pes_.reserve(config_.size * config_.size);
  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      ProcessingElementConfig pe_config;
      pe_config.name = config_.name + "_pe_" + std::to_string(row) + "_" +
                       std::to_string(col);
      pe_config.mac_latency_cycles = config_.mac_latency_cycles;
      pe_config.dataflow_mode = config_.dataflow_mode;
      pe_config.activation = config_.activation;
      pe_config.weight = config_.weight;
      pe_config.accumulator = config_.accumulator;
      const std::string module_name_string =
          std::string(name()) + "_pe_" + std::to_string(row) + "_" +
          std::to_string(col);
      pes_.push_back(std::make_unique<ProcessingElement>(
          module_name_string.c_str(), pe_config));
    }
  }

  if (config_.bias.enabled &&
      config_.bias.placement == "after_systolic_array") {
    const std::string module_name_string = std::string(name()) + "_bias_adder";
    bias_pipeline_ = std::make_unique<BiasAdderPipeline>(
        module_name_string.c_str(), config_.bias);
  }
  if (config_.requantization.enabled &&
      config_.requantization.placement == "after_systolic_array") {
    const std::string module_name_string =
        std::string(name()) + "_requantization";
    requantization_pipeline_ = std::make_unique<RequantizationPipeline>(
        module_name_string.c_str(), config_.requantization);
  }
}

const std::string& SystolicArray::name() const { return config_.name; }

const SystolicArrayConfig& SystolicArray::config() const { return config_; }

std::uint64_t SystolicArray::size() const { return config_.size; }

std::uint64_t SystolicArray::component_count() const {
  return pes_.size() + (bias_pipeline_ ? 1 : 0) +
         (requantization_pipeline_ ? 1 : 0);
}

SystolicArrayResult SystolicArray::run_matrix_multiply(
    const PEInputMatrix& activations, const PEInputMatrix& weights,
    bool trace_enabled) {
  return run_matrix_multiply_stream({activations}, {weights}, trace_enabled);
}

SystolicArrayResult SystolicArray::run_matrix_multiply_stream(
    const PEInputMatrixBatch& activation_batches,
    const PEInputMatrixBatch& weight_batches, bool trace_enabled) {
  if (activation_batches.empty()) {
    throw std::invalid_argument("systolic array stream must contain a batch");
  }
  if (activation_batches.size() != weight_batches.size()) {
    throw std::invalid_argument(
        "activation and weight stream batch counts must match");
  }

  std::vector<SystolicTraceEvent> trace;
  std::vector<std::vector<std::vector<PEMacResult>>> output_batches;
  output_batches.reserve(activation_batches.size());
  Cycle completion_cycle = current_cycle_from_systemc();
  Cycle drain_cycle = completion_cycle;
  Cycle next_base_cycle = completion_cycle;
  Cycle next_weight_load_base = completion_cycle;
  Cycle next_compute_base = completion_cycle;

  for (std::uint64_t batch = 0; batch < activation_batches.size(); ++batch) {
    validate_matrix_shape(activation_batches[batch], "activations");
    validate_matrix_shape(weight_batches[batch], "weights");
    const auto activation_values = convert_matrix(
        activation_batches[batch], config_.activation.format, "activation");
    const auto weight_values =
        convert_matrix(weight_batches[batch], config_.weight.format, "weight");
    auto accumulators = zero_accumulators();

    if (config_.dataflow_mode == PEDataflowMode::WeightStationary) {
      const Cycle weight_load_cycles = 2 * config_.size - 1;
      const bool use_shadow_weights = batch > 0;
      if (batch == 0) {
        next_compute_base = next_weight_load_base + weight_load_cycles;
      } else if (next_compute_base < next_weight_load_base + weight_load_cycles) {
        next_compute_base = next_weight_load_base + weight_load_cycles;
      }
      completion_cycle = weight_stationary_schedule(
          activation_values, weight_values, accumulators, trace, trace_enabled,
          next_weight_load_base, next_compute_base, batch, use_shadow_weights);
      output_batches.push_back(
          make_output_matrix(accumulators,
                             completion_cycle + output_pipeline_latency()));
      drain_cycle =
          std::max(drain_cycle, completion_cycle + output_pipeline_latency());
      next_weight_load_base = next_compute_base;
      next_compute_base = completion_cycle + 1;
    } else {
      completion_cycle = output_stationary_schedule(
          activation_values, weight_values, accumulators, trace, trace_enabled,
          next_base_cycle, batch);
      output_batches.push_back(
          make_output_matrix(accumulators,
                             completion_cycle + output_pipeline_latency()));
      drain_cycle =
          std::max(drain_cycle, completion_cycle + output_pipeline_latency());
      next_base_cycle = completion_cycle + 1;
    }
  }

  const Cycle current_cycle = current_cycle_from_systemc();
  if (drain_cycle > current_cycle) {
    sc_core::sc_start(sc_core::sc_time(
        static_cast<double>(drain_cycle - current_cycle), sc_core::SC_NS));
  }
  last_completion_cycle_ = drain_cycle;
  const std::uint64_t operations_per_batch =
      config_.size * config_.size * config_.size;
  operation_count_ += operations_per_batch * activation_batches.size();

  SystolicArrayResult result;
  result.output_batches = std::move(output_batches);
  result.outputs = result.output_batches.back();
  result.trace = std::move(trace);
  result.current_cycle = last_completion_cycle_;
  result.operation_count = operations_per_batch * activation_batches.size();
  result.dataflow_mode = config_.dataflow_mode;
  result.size = config_.size;
  return result;
}

void SystolicArray::validate_config() const {
  if (config_.name.empty()) {
    throw std::invalid_argument("systolic array name must be non-empty");
  }
  if (config_.size == 0) {
    throw std::invalid_argument("systolic array size must be positive");
  }
  validate_numeric_format(config_.activation.format);
  validate_numeric_format(config_.weight.format);
  validate_numeric_format(config_.accumulator);
  validate_numeric_format(config_.bias.format);
  validate_numeric_format(config_.requantization.target);
  const auto validate_register = [this](const PipelineRegister& reg,
                                        const char* prefix, const char* name) {
    if (reg.values.empty()) {
      throw std::invalid_argument(std::string(prefix) + " " + name +
                                  " register must contain a value");
    }
    if (reg.per_column && reg.values.size() != config_.size) {
      throw std::invalid_argument(std::string(prefix) + " " + name +
                                  " vector length must match systolic array size");
    }
  };
  validate_register(config_.bias.bias, "bias", "bias");
  validate_register(config_.requantization.scale_multiplier,
                    "requantization", "scale_multiplier");
  validate_register(config_.requantization.shift, "requantization", "shift");
  if (config_.bias.enabled) {
    if (config_.accumulator.kind != NumericFormatKind::Float &&
        config_.bias.format.kind == NumericFormatKind::Float) {
      throw std::invalid_argument(
          "integer accumulator requires an integer bias format");
    }
    if (config_.bias.format.kind != NumericFormatKind::Float) {
      const std::uint64_t operand_bits =
          std::max(config_.activation.format.bits, config_.weight.format.bits);
      if (config_.bias.format.bits <= operand_bits) {
        throw std::invalid_argument(
            "bias format bits must be greater than activation/weight bits");
      }
    }
  }
  for (std::uint64_t col = 0; col < config_.size; ++col) {
    const std::int64_t shift =
        integer_register_value(config_.requantization.shift, col, "shift");
    if (shift < 0) {
      throw std::invalid_argument("requantization shift must be non-negative");
    }
  }
  if (config_.requantization.enabled &&
      config_.requantization.target.kind != NumericFormatKind::Float &&
      config_.accumulator.kind == NumericFormatKind::Float) {
    throw std::invalid_argument(
        "integer requantization target requires an integer accumulator");
  }
}

void SystolicArray::validate_matrix_shape(const PEInputMatrix& matrix,
                                          const char* matrix_name) const {
  if (matrix.size() != config_.size) {
    throw std::invalid_argument(std::string(matrix_name) +
                                " matrix must match systolic array size");
  }
  for (const auto& row : matrix) {
    if (row.size() != config_.size) {
      throw std::invalid_argument(std::string(matrix_name) +
                                  " matrix must be square");
    }
  }
}

PEValue SystolicArray::make_value(const NumericFormatConfig& format,
                                  const PEInputValue& input,
                                  const char* operand_name) const {
  if (format.kind == NumericFormatKind::Float) {
    const double value =
        input.is_float ? input.floating_value
                       : static_cast<double>(input.integer_value);
    return PEValue::floating(format, value);
  }
  if (input.is_float) {
    throw std::invalid_argument(std::string("integer systolic array ") +
                                operand_name +
                                " input must be a Python int");
  }
  return PEValue::integer(format, input.integer_value);
}

PEValue SystolicArray::multiply_accumulate(const PEValue& accumulator,
                                           const PEValue& activation,
                                           const PEValue& weight) const {
  if (accumulator.is_unknown() || activation.is_unknown() ||
      weight.is_unknown()) {
    return PEValue::unknown(config_.accumulator);
  }

  const double value = accumulator.as_behavioral_double() +
                       activation.as_behavioral_double() *
                           weight.as_behavioral_double();
  if (config_.accumulator.kind == NumericFormatKind::Float) {
    return PEValue::floating(config_.accumulator, value);
  }
  return PEValue::integer(config_.accumulator,
                          static_cast<std::int64_t>(std::llround(value)));
}

PEMacResult SystolicArray::make_result(const PEValue& value,
                                       Cycle current_cycle,
                                       std::uint64_t mac_count) const {
  PEMacResult result;
  result.format = value.format();
  result.unknown = value.is_unknown();
  result.current_cycle = current_cycle;
  result.mac_count = mac_count;
  if (!result.unknown) {
    if (value.is_float()) {
      result.floating_value = value.floating_value();
    } else {
      result.integer_value = value.integer_value();
    }
  }
  return result;
}

Cycle SystolicArray::output_pipeline_latency() const {
  return (bias_pipeline_ ? bias_pipeline_->latency() : 0) +
         (requantization_pipeline_ ? requantization_pipeline_->latency() : 0);
}

bool SystolicArray::has_output_pipeline() const {
  return output_pipeline_latency() > 0;
}

PEValue SystolicArray::process_output(const PEValue& accumulator,
                                      std::uint64_t col) const {
  PEValue value = accumulator;
  if (bias_pipeline_) {
    value = bias_pipeline_->apply(value, col);
  }
  if (requantization_pipeline_) {
    value = requantization_pipeline_->apply(value, col);
  }
  return value;
}

void SystolicArray::add_output_pipeline_trace(
    std::vector<SystolicTraceEvent>& trace, Cycle cycle, std::uint64_t batch,
    std::int64_t row, std::int64_t col, const PEValue& accumulator) const {
  Cycle stage_cycle = cycle;
  PEValue value = accumulator;
  const std::uint64_t column = static_cast<std::uint64_t>(col);
  if (bias_pipeline_) {
    ++stage_cycle;
    value = bias_pipeline_->apply(value, column);
    add_bias_trace_value(trace, stage_cycle, batch, row, col, value);
  }
  if (requantization_pipeline_) {
    for (const OutputPipelineStage& stage :
         requantization_pipeline_->trace_stages(value, column)) {
      ++stage_cycle;
      add_requant_trace_value(trace, stage_cycle, stage, batch, row, col);
    }
  }
}

void SystolicArray::add_trace_value(std::vector<SystolicTraceEvent>& trace,
                                    Cycle cycle, std::string event,
                                    std::uint64_t batch, std::int64_t row,
                                    std::int64_t col, std::int64_t k,
                                    const PEValue& value) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = std::move(event);
  trace_event.array = config_.name;
  trace_event.batch = batch;
  trace_event.row = row;
  trace_event.col = col;
  trace_event.k = k;
  trace_event.has_value = !value.is_unknown();
  if (trace_event.has_value) {
    trace_event.value = value_as_double(value);
    trace_event.value_is_float = value.is_float();
  }
  trace.push_back(std::move(trace_event));
}

void SystolicArray::add_bias_trace_value(
    std::vector<SystolicTraceEvent>& trace, Cycle cycle, std::uint64_t batch,
    std::int64_t row, std::int64_t col, const PEValue& value) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = "bias_add";
  trace_event.array = config_.name;
  trace_event.batch = batch;
  trace_event.row = row;
  trace_event.col = col;
  trace_event.k = -1;
  trace_event.has_value = !value.is_unknown();
  if (trace_event.has_value) {
    trace_event.value = value_as_double(value);
    trace_event.value_is_float = value.is_float();
  }
  trace_event.has_bias_metadata = true;
  trace_event.bias =
      register_value(config_.bias.bias, static_cast<std::uint64_t>(col),
                     "bias");
  trace_event.bias_format_kind =
      numeric_format_kind_name(config_.bias.format.kind);
  trace_event.bias_format_bits = config_.bias.format.bits;
  trace_event.placement = config_.bias.placement;
  trace.push_back(std::move(trace_event));
}

void SystolicArray::add_requant_trace_value(
    std::vector<SystolicTraceEvent>& trace, Cycle cycle,
    const OutputPipelineStage& stage, std::uint64_t batch, std::int64_t row,
    std::int64_t col) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = stage.event;
  trace_event.array = config_.name;
  trace_event.batch = batch;
  trace_event.row = row;
  trace_event.col = col;
  trace_event.k = -1;
  trace_event.has_value = !stage.value.is_unknown();
  if (trace_event.has_value) {
    trace_event.value = value_as_double(stage.value);
    trace_event.value_is_float = stage.value.is_float();
  }
  trace_event.has_requantization_metadata = true;
  trace_event.scale_multiplier = stage.scale_multiplier;
  trace_event.shift = stage.shift;
  trace_event.placement = config_.requantization.placement;
  trace_event.target_kind = stage.target_kind;
  trace_event.target_bits = stage.target_bits;
  trace.push_back(std::move(trace_event));
}

void SystolicArray::add_trace_no_value(std::vector<SystolicTraceEvent>& trace,
                                       Cycle cycle, std::string event,
                                       std::uint64_t batch, std::int64_t row,
                                       std::int64_t col, std::int64_t k) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = std::move(event);
  trace_event.array = config_.name;
  trace_event.batch = batch;
  trace_event.row = row;
  trace_event.col = col;
  trace_event.k = k;
  trace.push_back(std::move(trace_event));
}

Cycle SystolicArray::current_cycle_from_systemc() const {
  return static_cast<Cycle>(
      std::llround(sc_core::sc_time_stamp().to_seconds() / 1e-9));
}

std::vector<std::vector<PEValue>> SystolicArray::convert_matrix(
    const PEInputMatrix& matrix, const NumericFormatConfig& format,
    const char* operand_name) const {
  std::vector<std::vector<PEValue>> values;
  values.reserve(config_.size);
  for (std::uint64_t row = 0; row < config_.size; ++row) {
    std::vector<PEValue> value_row;
    value_row.reserve(config_.size);
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      value_row.push_back(make_value(format, matrix[row][col], operand_name));
    }
    values.push_back(std::move(value_row));
  }
  return values;
}

std::vector<std::vector<PEValue>> SystolicArray::zero_accumulators() const {
  std::vector<std::vector<PEValue>> accumulators;
  accumulators.reserve(config_.size);
  for (std::uint64_t row = 0; row < config_.size; ++row) {
    std::vector<PEValue> accumulator_row;
    accumulator_row.reserve(config_.size);
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      accumulator_row.push_back(zero_value_for_format(config_.accumulator));
    }
    accumulators.push_back(std::move(accumulator_row));
  }
  return accumulators;
}

std::vector<std::vector<PEMacResult>> SystolicArray::make_output_matrix(
    const std::vector<std::vector<PEValue>>& accumulators,
    Cycle current_cycle) const {
  std::vector<std::vector<PEMacResult>> outputs;
  outputs.reserve(config_.size);
  for (std::uint64_t row = 0; row < config_.size; ++row) {
    std::vector<PEMacResult> output_row;
    output_row.reserve(config_.size);
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      output_row.push_back(make_result(process_output(accumulators[row][col],
                                                     col),
                                       current_cycle, config_.size));
    }
    outputs.push_back(std::move(output_row));
  }
  return outputs;
}

Cycle SystolicArray::output_stationary_schedule(
    const std::vector<std::vector<PEValue>>& activations,
    const std::vector<std::vector<PEValue>>& weights,
    std::vector<std::vector<PEValue>>& accumulators,
    std::vector<SystolicTraceEvent>& trace, bool trace_enabled,
    Cycle base_cycle, std::uint64_t batch) const {
  Cycle last_cycle = base_cycle;
  std::vector<StreamFifo> activation_fifos(config_.size);
  std::vector<StreamFifo> weight_fifos(config_.size);

  for (std::uint64_t k = 0; k < config_.size; ++k) {
    for (std::uint64_t row = 0; row < config_.size; ++row) {
      const Cycle inject_cycle = base_cycle + row + k;
      activation_fifos[row].emplace_back(inject_cycle, row, -1, k,
                                         activations[row][k]);
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "activation_fifo_push", batch, row,
                        -1, k, activations[row][k]);
        const StreamItem item = activation_fifos[row].front();
        activation_fifos[row].pop_front();
        add_trace_value(trace, item.cycle, "activation_fifo_pop", batch,
                        item.row, item.col, item.k, item.value);
        add_trace_value(trace, inject_cycle, "inject_activation", batch, row, -1,
                        k, activations[row][k]);
      } else {
        activation_fifos[row].pop_front();
      }
    }
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      const Cycle inject_cycle = base_cycle + col + k;
      weight_fifos[col].emplace_back(inject_cycle, -1, col, k,
                                     weights[k][col]);
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "weight_fifo_push", batch, -1,
                        col, k, weights[k][col]);
        const StreamItem item = weight_fifos[col].front();
        weight_fifos[col].pop_front();
        add_trace_value(trace, item.cycle, "weight_fifo_pop", batch, item.row,
                        item.col, item.k, item.value);
        add_trace_value(trace, inject_cycle, "inject_weight", batch, -1, col, k,
                        weights[k][col]);
      } else {
        weight_fifos[col].pop_front();
      }
    }
  }

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      for (std::uint64_t k = 0; k < config_.size; ++k) {
        const Cycle start_cycle = base_cycle + row + col + k;
        const Cycle commit_cycle = start_cycle + config_.mac_latency_cycles;
        if (trace_enabled) {
          add_trace_value(trace, start_cycle, "activation_arrive", batch, row, col,
                          k, activations[row][k]);
          add_trace_value(trace, start_cycle, "weight_arrive", batch, row, col, k,
                          weights[k][col]);
          add_trace_no_value(trace, start_cycle, "mac_start", batch, row, col, k);
        }
        accumulators[row][col] = multiply_accumulate(
            accumulators[row][col], activations[row][k], weights[k][col]);
        if (trace_enabled) {
          add_trace_value(trace, commit_cycle, "mac_commit", batch, row, col, k,
                          accumulators[row][col]);
        }
        if (commit_cycle > last_cycle) {
          last_cycle = commit_cycle;
        }
      }
      const Cycle emit_cycle =
          base_cycle + row + col + (config_.size - 1) +
          config_.mac_latency_cycles;
      const Cycle final_emit_cycle = emit_cycle + output_pipeline_latency();
      if (trace_enabled) {
        if (has_output_pipeline()) {
          add_trace_value(trace, emit_cycle, "pre_requant_output", batch, row,
                          col, -1, accumulators[row][col]);
          add_output_pipeline_trace(trace, emit_cycle, batch, row, col,
                                    accumulators[row][col]);
        }
        add_trace_value(trace, final_emit_cycle, "output_emit", batch, row, col,
                        -1, process_output(accumulators[row][col], col));
      }
      if (emit_cycle > last_cycle) {
        last_cycle = emit_cycle;
      }
    }
  }

  return last_cycle;
}

Cycle SystolicArray::weight_stationary_schedule(
    const std::vector<std::vector<PEValue>>& activations,
    const std::vector<std::vector<PEValue>>& weights,
    std::vector<std::vector<PEValue>>& accumulators,
    std::vector<SystolicTraceEvent>& trace, bool trace_enabled,
    Cycle load_base_cycle, Cycle compute_base_cycle, std::uint64_t batch,
    bool use_shadow_weights) const {
  Cycle last_cycle = compute_base_cycle;
  const char* load_event =
      use_shadow_weights ? "shadow_weight_load" : "weight_load";
  const char* activate_event =
      use_shadow_weights ? "shadow_weight_activate" : "weight_activate";
  std::vector<StreamFifo> activation_fifos(config_.size);
  std::vector<StreamFifo> weight_fifos(config_.size * config_.size);

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      const Cycle load_cycle = load_base_cycle + row + col;
      StreamFifo& weight_fifo = weight_fifos[row * config_.size + col];
      weight_fifo.emplace_back(load_cycle, row, col, row, weights[row][col]);
      if (trace_enabled) {
        add_trace_value(trace, load_cycle, "weight_fifo_push", batch, row, col,
                        row, weights[row][col]);
        const StreamItem item = weight_fifo.front();
        weight_fifo.pop_front();
        add_trace_value(trace, compute_base_cycle, "weight_fifo_pop", batch,
                        item.row, item.col, item.k, item.value);
        add_trace_value(trace, compute_base_cycle, load_event, batch, row, col, row,
                        weights[row][col]);
      } else {
        weight_fifo.pop_front();
      }
    }
  }
  if (trace_enabled) {
    add_trace_no_value(trace, compute_base_cycle, activate_event, batch, -1, -1,
                       -1);
  }

  for (std::uint64_t k = 0; k < config_.size; ++k) {
    for (std::uint64_t row = 0; row < config_.size; ++row) {
      const Cycle inject_cycle = compute_base_cycle + row + k;
      activation_fifos[row].emplace_back(inject_cycle, row, -1, k,
                                         activations[row][k]);
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "activation_fifo_push", batch, row,
                        -1, k, activations[row][k]);
        const StreamItem item = activation_fifos[row].front();
        activation_fifos[row].pop_front();
        add_trace_value(trace, item.cycle, "activation_fifo_pop", batch,
                        item.row, item.col, item.k, item.value);
        add_trace_value(trace, inject_cycle, "inject_activation", batch, row, -1,
                        k, activations[row][k]);
      } else {
        activation_fifos[row].pop_front();
      }
    }
  }

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      for (std::uint64_t k = 0; k < config_.size; ++k) {
        const Cycle start_cycle = compute_base_cycle + row + col + k;
        const Cycle commit_cycle = start_cycle + config_.mac_latency_cycles;
        if (trace_enabled) {
          add_trace_value(trace, start_cycle, "activation_arrive", batch, row, col,
                          k, activations[row][k]);
          add_trace_value(trace, start_cycle, "stationary_weight", batch, k, col, k,
                          weights[k][col]);
          add_trace_no_value(trace, start_cycle, "mac_start", batch, row, col, k);
        }
        accumulators[row][col] = multiply_accumulate(
            accumulators[row][col], activations[row][k], weights[k][col]);
        if (trace_enabled) {
          add_trace_value(trace, commit_cycle, "partial_sum_commit", batch, row, col,
                          k, accumulators[row][col]);
        }
        if (commit_cycle > last_cycle) {
          last_cycle = commit_cycle;
        }
      }
      const Cycle emit_cycle =
          compute_base_cycle + row + col + (config_.size - 1) +
          config_.mac_latency_cycles + (config_.size - 1 - row);
      const Cycle final_emit_cycle = emit_cycle + output_pipeline_latency();
      if (trace_enabled) {
        if (has_output_pipeline()) {
          add_trace_value(trace, emit_cycle, "pre_requant_output", batch, row,
                          col, -1, accumulators[row][col]);
          add_output_pipeline_trace(trace, emit_cycle, batch, row, col,
                                    accumulators[row][col]);
        }
        add_trace_value(trace, final_emit_cycle, "output_emit", batch, row, col,
                        -1, process_output(accumulators[row][col], col));
      }
      if (emit_cycle > last_cycle) {
        last_cycle = emit_cycle;
      }
    }
  }

  return last_cycle;
}

}  // namespace npu_sim
