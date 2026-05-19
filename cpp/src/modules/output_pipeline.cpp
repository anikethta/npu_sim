#include "npu_sim/modules/output_pipeline.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace npu_sim {

namespace {

std::string numeric_format_kind_name(NumericFormatKind kind) {
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

__int128 min_integer_value(const NumericFormatConfig& format) {
  if (format.kind == NumericFormatKind::SignedInt) {
    return -(__int128{1} << (format.bits - 1));
  }
  if (format.kind == NumericFormatKind::UnsignedInt) {
    return 0;
  }
  throw std::invalid_argument("integer range requires integer format");
}

__int128 max_integer_value(const NumericFormatConfig& format) {
  if (format.kind == NumericFormatKind::SignedInt) {
    return (__int128{1} << (format.bits - 1)) - 1;
  }
  if (format.kind == NumericFormatKind::UnsignedInt) {
    return (__int128{1} << format.bits) - 1;
  }
  throw std::invalid_argument("integer range requires integer format");
}

std::int64_t saturate_to_int64(__int128 value) {
  const __int128 min_value =
      static_cast<__int128>(std::numeric_limits<std::int64_t>::min());
  const __int128 max_value =
      static_cast<__int128>(std::numeric_limits<std::int64_t>::max());
  value = std::min(std::max(value, min_value), max_value);
  return static_cast<std::int64_t>(value);
}

__int128 arithmetic_right_shift(__int128 value, std::int64_t shift) {
  if (shift <= 0) {
    return value;
  }
  if (shift >= 127) {
    return value < 0 ? -1 : 0;
  }
  if (value >= 0) {
    return value >> shift;
  }
  const __int128 divisor = __int128{1} << shift;
  return -(((-value) + divisor - 1) >> shift);
}

PEValue integer_stage_value(const NumericFormatConfig& format,
                            __int128 value) {
  value = std::min(std::max(value, min_integer_value(format)),
                   max_integer_value(format));
  return PEValue::integer(format, saturate_to_int64(value));
}

}  // namespace

double register_value(const PipelineRegister& reg, std::uint64_t col,
                      const char* name) {
  if (reg.values.empty()) {
    throw std::invalid_argument(std::string(name) +
                                " register must contain a value");
  }
  const std::size_t index =
      reg.per_column ? static_cast<std::size_t>(col) : std::size_t{0};
  if (index >= reg.values.size()) {
    throw std::invalid_argument(std::string(name) +
                                " register does not contain this column");
  }
  return reg.values[index];
}

std::int64_t integer_register_value(const PipelineRegister& reg,
                                    std::uint64_t col, const char* name) {
  const double value = register_value(reg, col, name);
  if (!std::isfinite(value) || std::floor(value) != value) {
    throw std::invalid_argument(std::string(name) +
                                " register value must be an integer");
  }
  if (value < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
      value > static_cast<double>(std::numeric_limits<std::int64_t>::max())) {
    throw std::out_of_range(std::string(name) +
                            " register value is out of int64 range");
  }
  return static_cast<std::int64_t>(value);
}

BiasAdderPipeline::BiasAdderPipeline(sc_core::sc_module_name module_name,
                                     BiasAdderConfig config)
    : sc_core::sc_module(module_name), config_(std::move(config)) {}

const BiasAdderConfig& BiasAdderPipeline::config() const { return config_; }

Cycle BiasAdderPipeline::latency() const { return config_.enabled ? 1 : 0; }

PEValue BiasAdderPipeline::apply(const PEValue& accumulator,
                                 std::uint64_t col) const {
  if (!config_.enabled || accumulator.is_unknown()) {
    return accumulator;
  }
  if (accumulator.is_float()) {
    return PEValue::floating(
        accumulator.format(),
        accumulator.floating_value() + register_value(config_.bias, col, "bias"));
  }
  const __int128 value = static_cast<__int128>(accumulator.integer_value()) +
                         static_cast<__int128>(
                             integer_register_value(config_.bias, col, "bias"));
  return PEValue::integer(accumulator.format(), saturate_to_int64(value));
}

OutputPipelineStage BiasAdderPipeline::trace_stage(
    const PEValue& accumulator, std::uint64_t col) const {
  OutputPipelineStage stage{"bias_add", apply(accumulator, col)};
  return stage;
}

RequantizationPipeline::RequantizationPipeline(
    sc_core::sc_module_name module_name, RequantizationConfig config)
    : sc_core::sc_module(module_name), config_(std::move(config)) {}

const RequantizationConfig& RequantizationPipeline::config() const {
  return config_;
}

Cycle RequantizationPipeline::latency() const {
  return config_.enabled ? 4 : 0;
}

PEValue RequantizationPipeline::apply(const PEValue& value,
                                      std::uint64_t col) const {
  if (!config_.enabled || value.is_unknown()) {
    return value;
  }

  const NumericFormatConfig& target = config_.target;
  const std::int64_t shift =
      integer_register_value(config_.shift, col, "shift");
  if (target.kind == NumericFormatKind::Float) {
    double output = value.as_behavioral_double();
    output *= register_value(config_.scale_multiplier, col, "scale_multiplier");
    if (shift > 0) {
      output /= std::pow(2.0, static_cast<double>(shift));
    }
    return PEValue::floating(target, output);
  }

  __int128 output = static_cast<__int128>(value.integer_value());
  output *= static_cast<__int128>(
      integer_register_value(config_.scale_multiplier, col, "scale_multiplier"));
  if (config_.rounding && shift > 0) {
    output += __int128{1} << (shift - 1);
  }
  output = arithmetic_right_shift(output, shift);
  output = std::min(std::max(output, min_integer_value(target)),
                    max_integer_value(target));
  return PEValue::integer(target, saturate_to_int64(output));
}

std::vector<OutputPipelineStage> RequantizationPipeline::trace_stages(
    const PEValue& value, std::uint64_t col) const {
  std::vector<OutputPipelineStage> stages;
  if (!config_.enabled) {
    return stages;
  }

  const NumericFormatConfig& target = config_.target;
  const double scale_multiplier =
      register_value(config_.scale_multiplier, col, "scale_multiplier");
  const std::int64_t shift =
      integer_register_value(config_.shift, col, "shift");

  const auto make_stage = [&](std::string event, PEValue stage_value) {
    OutputPipelineStage stage{std::move(event), std::move(stage_value)};
    stage.scale_multiplier = scale_multiplier;
    stage.shift = shift;
    stage.target_kind = numeric_format_kind_name(target.kind);
    stage.target_bits = target.bits;
    stages.push_back(std::move(stage));
  };

  if (target.kind == NumericFormatKind::Float) {
    double output = value.as_behavioral_double() * scale_multiplier;
    make_stage("requant_multiply", PEValue::floating(target, output));
    make_stage("requant_round", PEValue::floating(target, output));
    if (shift > 0) {
      output /= std::pow(2.0, static_cast<double>(shift));
    }
    make_stage("requant_shift", PEValue::floating(target, output));
    make_stage("requant_saturate", PEValue::floating(target, output));
    return stages;
  }

  const NumericFormatConfig intermediate_format{
      NumericFormatKind::SignedInt,
      63,
      0,
      0,
  };
  __int128 output = static_cast<__int128>(value.integer_value());
  output *= static_cast<__int128>(
      integer_register_value(config_.scale_multiplier, col, "scale_multiplier"));
  make_stage("requant_multiply",
             integer_stage_value(intermediate_format, output));
  if (config_.rounding && shift > 0) {
    output += __int128{1} << (shift - 1);
  }
  make_stage("requant_round", integer_stage_value(intermediate_format, output));
  output = arithmetic_right_shift(output, shift);
  make_stage("requant_shift", integer_stage_value(intermediate_format, output));
  output = std::min(std::max(output, min_integer_value(target)),
                    max_integer_value(target));
  make_stage("requant_saturate",
             PEValue::integer(target, saturate_to_int64(output)));
  return stages;
}

}  // namespace npu_sim
