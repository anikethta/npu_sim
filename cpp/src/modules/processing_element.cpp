#include "npu_sim/modules/processing_element.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
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

void validate_integer_range(const NumericFormatConfig& format,
                            std::int64_t value) {
  validate_numeric_format(format);
  if (format.kind == NumericFormatKind::SignedInt) {
    const std::int64_t min_value = -(std::int64_t{1} << (format.bits - 1));
    const std::int64_t max_value = (std::int64_t{1} << (format.bits - 1)) - 1;
    if (value < min_value || value > max_value) {
      throw std::out_of_range("signed integer PE value is out of range");
    }
    return;
  }

  if (format.kind == NumericFormatKind::UnsignedInt) {
    const std::uint64_t max_value =
        (format.bits == 63) ? (std::uint64_t{1} << 63) - 1
                            : (std::uint64_t{1} << format.bits) - 1;
    if (value < 0 || static_cast<std::uint64_t>(value) > max_value) {
      throw std::out_of_range("unsigned integer PE value is out of range");
    }
    return;
  }

  throw std::invalid_argument("integer PE value requires integer format");
}

PEValue zero_value_for_format(const NumericFormatConfig& format) {
  if (format.kind == NumericFormatKind::Float) {
    return PEValue::floating(format, 0.0);
  }
  return PEValue::integer(format, 0);
}

}  // namespace

PEValue::PEValue(NumericFormatConfig format, bool unknown,
                 std::int64_t integer_value, double floating_value)
    : format_(format),
      unknown_(unknown),
      integer_value_(integer_value),
      floating_value_(floating_value) {}

PEValue PEValue::integer(NumericFormatConfig format, std::int64_t value) {
  validate_integer_range(format, value);
  return PEValue(format, false, value, static_cast<double>(value));
}

PEValue PEValue::floating(NumericFormatConfig format, double value) {
  validate_numeric_format(format);
  if (format.kind != NumericFormatKind::Float) {
    throw std::invalid_argument("floating PE value requires float format");
  }
  if (!std::isfinite(value)) {
    throw std::invalid_argument("floating PE value must be finite");
  }
  return PEValue(format, false, 0, value);
}

PEValue PEValue::unknown(NumericFormatConfig format) {
  validate_numeric_format(format);
  return PEValue(format, true, 0, 0.0);
}

const NumericFormatConfig& PEValue::format() const { return format_; }

bool PEValue::is_unknown() const { return unknown_; }

bool PEValue::is_float() const {
  return format_.kind == NumericFormatKind::Float;
}

std::int64_t PEValue::integer_value() const {
  if (unknown_) {
    throw std::logic_error("unknown PE value has no integer value");
  }
  if (is_float()) {
    throw std::logic_error("floating PE value has no integer value");
  }
  return integer_value_;
}

double PEValue::floating_value() const {
  if (unknown_) {
    throw std::logic_error("unknown PE value has no floating value");
  }
  if (!is_float()) {
    throw std::logic_error("integer PE value has no floating value");
  }
  return floating_value_;
}

double PEValue::as_behavioral_double() const {
  if (unknown_) {
    throw std::logic_error("unknown PE value has no behavioral value");
  }
  return is_float() ? floating_value_ : static_cast<double>(integer_value_);
}

ProcessingElement::ProcessingElement(sc_core::sc_module_name module_name,
                                     ProcessingElementConfig config)
    : sc_core::sc_module(module_name),
      config_(std::move(config)),
      accumulator_(zero_value_for_format(config_.accumulator)) {
  validate_config();
}

const std::string& ProcessingElement::name() const { return config_.name; }

const ProcessingElementConfig& ProcessingElement::config() const {
  return config_;
}

Cycle ProcessingElement::mac_latency_cycles() const {
  return config_.mac_latency_cycles;
}

PEDataflowMode ProcessingElement::dataflow_mode() const {
  return config_.dataflow_mode;
}

std::uint64_t ProcessingElement::mac_count() const { return mac_count_; }

Cycle ProcessingElement::last_completion_cycle() const {
  return last_completion_cycle_;
}

const PEValue& ProcessingElement::accumulator() const { return accumulator_; }

PEValue ProcessingElement::mac(const PEValue& activation,
                               const PEValue& weight) {
  validate_operand(activation, config_.activation.format, "activation");
  validate_operand(weight, config_.weight.format, "weight");

  advance_cycles(config_.mac_latency_cycles);
  ++mac_count_;

  if (activation.is_unknown() || weight.is_unknown() ||
      accumulator_.is_unknown()) {
    accumulator_ = PEValue::unknown(config_.accumulator);
    return accumulator_;
  }

  const double result = accumulator_.as_behavioral_double() +
                        activation.as_behavioral_double() *
                            weight.as_behavioral_double();

  if (config_.accumulator.kind == NumericFormatKind::Float) {
    accumulator_ = PEValue::floating(config_.accumulator, result);
  } else {
    accumulator_ = PEValue::integer(
        config_.accumulator, static_cast<std::int64_t>(std::llround(result)));
  }

  return accumulator_;
}

PEValue ProcessingElement::pass_activation(const PEValue& value) const {
  return value;
}

PEValue ProcessingElement::pass_weight(const PEValue& value) const {
  return value;
}

PEValue ProcessingElement::pass_result(const PEValue& value) const {
  return value;
}

void ProcessingElement::advance_cycles(Cycle cycles) {
  if (cycles > 0) {
    sc_core::sc_start(sc_core::sc_time(static_cast<double>(cycles),
                                       sc_core::SC_NS));
  }
  last_completion_cycle_ =
      static_cast<Cycle>(
          std::llround(sc_core::sc_time_stamp().to_seconds() / 1e-9));
}

void ProcessingElement::validate_config() const {
  if (config_.name.empty()) {
    throw std::invalid_argument("processing element name must be non-empty");
  }
  validate_numeric_format(config_.activation.format);
  validate_numeric_format(config_.weight.format);
  validate_numeric_format(config_.accumulator);
}

void ProcessingElement::validate_operand(
    const PEValue& value, const NumericFormatConfig& expected_format,
    const char* operand_name) const {
  if (!same_numeric_format(value.format(), expected_format)) {
    throw std::invalid_argument(std::string("PE ") + operand_name +
                                " format does not match config");
  }
}

bool same_numeric_format(const NumericFormatConfig& lhs,
                         const NumericFormatConfig& rhs) {
  return lhs.kind == rhs.kind && lhs.bits == rhs.bits &&
         lhs.exponent_bits == rhs.exponent_bits &&
         lhs.mantissa_bits == rhs.mantissa_bits;
}

}  // namespace npu_sim
