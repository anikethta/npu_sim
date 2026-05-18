#include "npu_sim/modules/systolic_array.hpp"

#include <cmath>
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
}

const std::string& SystolicArray::name() const { return config_.name; }

const SystolicArrayConfig& SystolicArray::config() const { return config_; }

std::uint64_t SystolicArray::size() const { return config_.size; }

std::uint64_t SystolicArray::component_count() const { return pes_.size(); }

SystolicArrayResult SystolicArray::run_matrix_multiply(
    const PEInputMatrix& activations, const PEInputMatrix& weights,
    bool trace_enabled) {
  validate_matrix_shape(activations, "activations");
  validate_matrix_shape(weights, "weights");

  std::vector<std::vector<PEValue>> activation_values;
  std::vector<std::vector<PEValue>> weight_values;
  std::vector<std::vector<PEValue>> accumulators;
  activation_values.reserve(config_.size);
  weight_values.reserve(config_.size);
  accumulators.reserve(config_.size);

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    std::vector<PEValue> activation_row;
    std::vector<PEValue> weight_row;
    std::vector<PEValue> accumulator_row;
    activation_row.reserve(config_.size);
    weight_row.reserve(config_.size);
    accumulator_row.reserve(config_.size);
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      activation_row.push_back(make_value(config_.activation.format,
                                          activations[row][col],
                                          "activation"));
      weight_row.push_back(
          make_value(config_.weight.format, weights[row][col], "weight"));
      accumulator_row.push_back(zero_value_for_format(config_.accumulator));
    }
    activation_values.push_back(std::move(activation_row));
    weight_values.push_back(std::move(weight_row));
    accumulators.push_back(std::move(accumulator_row));
  }

  std::vector<SystolicTraceEvent> trace;
  Cycle completion_cycle = current_cycle_from_systemc();
  if (config_.dataflow_mode == PEDataflowMode::WeightStationary) {
    completion_cycle = weight_stationary_schedule(
        activation_values, weight_values, accumulators, trace, trace_enabled);
  } else {
    completion_cycle = output_stationary_schedule(
        activation_values, weight_values, accumulators, trace, trace_enabled);
  }

  const Cycle current_cycle = current_cycle_from_systemc();
  if (completion_cycle > current_cycle) {
    sc_core::sc_start(sc_core::sc_time(
        static_cast<double>(completion_cycle - current_cycle), sc_core::SC_NS));
  }
  last_completion_cycle_ = completion_cycle;
  operation_count_ += config_.size * config_.size * config_.size;

  SystolicArrayResult result;
  result.outputs.reserve(config_.size);
  for (std::uint64_t row = 0; row < config_.size; ++row) {
    std::vector<PEMacResult> output_row;
    output_row.reserve(config_.size);
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      const Cycle emit_cycle =
          last_completion_cycle_;
      output_row.push_back(make_result(accumulators[row][col], emit_cycle,
                                       config_.size));
    }
    result.outputs.push_back(std::move(output_row));
  }
  result.trace = std::move(trace);
  result.current_cycle = last_completion_cycle_;
  result.operation_count = config_.size * config_.size * config_.size;
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

void SystolicArray::add_trace_value(std::vector<SystolicTraceEvent>& trace,
                                    Cycle cycle, std::string event,
                                    std::int64_t row, std::int64_t col,
                                    std::int64_t k,
                                    const PEValue& value) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = std::move(event);
  trace_event.array = config_.name;
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

void SystolicArray::add_trace_no_value(std::vector<SystolicTraceEvent>& trace,
                                       Cycle cycle, std::string event,
                                       std::int64_t row, std::int64_t col,
                                       std::int64_t k) const {
  SystolicTraceEvent trace_event;
  trace_event.cycle = cycle;
  trace_event.event = std::move(event);
  trace_event.array = config_.name;
  trace_event.row = row;
  trace_event.col = col;
  trace_event.k = k;
  trace.push_back(std::move(trace_event));
}

Cycle SystolicArray::current_cycle_from_systemc() const {
  return static_cast<Cycle>(
      std::llround(sc_core::sc_time_stamp().to_seconds() / 1e-9));
}

Cycle SystolicArray::output_stationary_schedule(
    const std::vector<std::vector<PEValue>>& activations,
    const std::vector<std::vector<PEValue>>& weights,
    std::vector<std::vector<PEValue>>& accumulators,
    std::vector<SystolicTraceEvent>& trace, bool trace_enabled) const {
  const Cycle base_cycle = current_cycle_from_systemc();
  Cycle last_cycle = base_cycle;

  for (std::uint64_t k = 0; k < config_.size; ++k) {
    for (std::uint64_t row = 0; row < config_.size; ++row) {
      const Cycle inject_cycle = base_cycle + row + k;
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "inject_activation", row, -1,
                        k, activations[row][k]);
      }
    }
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      const Cycle inject_cycle = base_cycle + col + k;
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "inject_weight", -1, col, k,
                        weights[k][col]);
      }
    }
  }

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      for (std::uint64_t k = 0; k < config_.size; ++k) {
        const Cycle start_cycle = base_cycle + row + col + k;
        const Cycle commit_cycle = start_cycle + config_.mac_latency_cycles;
        if (trace_enabled) {
          add_trace_value(trace, start_cycle, "activation_arrive", row, col,
                          k, activations[row][k]);
          add_trace_value(trace, start_cycle, "weight_arrive", row, col, k,
                          weights[k][col]);
          add_trace_no_value(trace, start_cycle, "mac_start", row, col, k);
        }
        accumulators[row][col] = multiply_accumulate(
            accumulators[row][col], activations[row][k], weights[k][col]);
        if (trace_enabled) {
          add_trace_value(trace, commit_cycle, "mac_commit", row, col, k,
                          accumulators[row][col]);
        }
        if (commit_cycle > last_cycle) {
          last_cycle = commit_cycle;
        }
      }
      const Cycle emit_cycle =
          base_cycle + row + col + (config_.size - 1) +
          config_.mac_latency_cycles;
      if (trace_enabled) {
        add_trace_value(trace, emit_cycle, "output_emit", row, col, -1,
                        accumulators[row][col]);
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
    std::vector<SystolicTraceEvent>& trace, bool trace_enabled) const {
  const Cycle base_cycle = current_cycle_from_systemc();
  const Cycle weight_load_cycles = 2 * config_.size - 1;
  const Cycle compute_base_cycle = base_cycle + weight_load_cycles;
  Cycle last_cycle = compute_base_cycle;

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      const Cycle load_cycle = base_cycle + row + col;
      if (trace_enabled) {
        add_trace_value(trace, load_cycle, "weight_load", row, col, row,
                        weights[row][col]);
      }
    }
  }

  for (std::uint64_t k = 0; k < config_.size; ++k) {
    for (std::uint64_t row = 0; row < config_.size; ++row) {
      const Cycle inject_cycle = compute_base_cycle + row + k;
      if (trace_enabled) {
        add_trace_value(trace, inject_cycle, "inject_activation", row, -1,
                        k, activations[row][k]);
      }
    }
  }

  for (std::uint64_t row = 0; row < config_.size; ++row) {
    for (std::uint64_t col = 0; col < config_.size; ++col) {
      for (std::uint64_t k = 0; k < config_.size; ++k) {
        const Cycle start_cycle = compute_base_cycle + row + col + k;
        const Cycle commit_cycle = start_cycle + config_.mac_latency_cycles;
        if (trace_enabled) {
          add_trace_value(trace, start_cycle, "activation_arrive", row, col,
                          k, activations[row][k]);
          add_trace_value(trace, start_cycle, "stationary_weight", k, col, k,
                          weights[k][col]);
          add_trace_no_value(trace, start_cycle, "mac_start", row, col, k);
        }
        accumulators[row][col] = multiply_accumulate(
            accumulators[row][col], activations[row][k], weights[k][col]);
        if (trace_enabled) {
          add_trace_value(trace, commit_cycle, "partial_sum_commit", row, col,
                          k, accumulators[row][col]);
        }
        if (commit_cycle > last_cycle) {
          last_cycle = commit_cycle;
        }
      }
      const Cycle emit_cycle =
          compute_base_cycle + row + col + (config_.size - 1) +
          config_.mac_latency_cycles + (config_.size - 1 - row);
      if (trace_enabled) {
        add_trace_value(trace, emit_cycle, "output_emit", row, col, -1,
                        accumulators[row][col]);
      }
      if (emit_cycle > last_cycle) {
        last_cycle = emit_cycle;
      }
    }
  }

  return last_cycle;
}

}  // namespace npu_sim
