#pragma once

#include <cstdint>
#include <string>

#include <systemc>

#include "npu_sim/types.hpp"

namespace npu_sim {

enum class PEFunctionKind {
  MAC,
};

enum class PEDataflowMode {
  OutputStationary,
  WeightStationary,
};

enum class NumericFormatKind {
  SignedInt,
  UnsignedInt,
  Float,
};

struct NumericFormatConfig {
  NumericFormatKind kind{NumericFormatKind::SignedInt};
  std::uint64_t bits{8};
  std::uint64_t exponent_bits{0};
  std::uint64_t mantissa_bits{0};
};

struct PEOperandConfig {
  NumericFormatConfig format;
};

struct ProcessingElementConfig {
  std::string name;
  Cycle mac_latency_cycles{1};
  PEDataflowMode dataflow_mode{PEDataflowMode::OutputStationary};
  PEOperandConfig activation;
  PEOperandConfig weight;
  NumericFormatConfig accumulator;
};

struct PEInputValue {
  bool is_float{false};
  std::int64_t integer_value{0};
  double floating_value{0.0};
};

struct PEMacResult {
  NumericFormatConfig format;
  bool unknown{false};
  std::int64_t integer_value{0};
  double floating_value{0.0};
  Cycle current_cycle{0};
  std::uint64_t mac_count{0};
};

class PEValue {
 public:
  static PEValue integer(NumericFormatConfig format, std::int64_t value);
  static PEValue floating(NumericFormatConfig format, double value);
  static PEValue unknown(NumericFormatConfig format);

  const NumericFormatConfig& format() const;
  bool is_unknown() const;
  bool is_float() const;
  std::int64_t integer_value() const;
  double floating_value() const;
  double as_behavioral_double() const;

 private:
  PEValue(NumericFormatConfig format, bool unknown, std::int64_t integer_value,
          double floating_value);

  NumericFormatConfig format_;
  bool unknown_{false};
  std::int64_t integer_value_{0};
  double floating_value_{0.0};
};

class ProcessingElement : public sc_core::sc_module {
 public:
  ProcessingElement(sc_core::sc_module_name module_name,
                    ProcessingElementConfig config);

  const std::string& name() const;
  const ProcessingElementConfig& config() const;
  Cycle mac_latency_cycles() const;
  PEDataflowMode dataflow_mode() const;
  std::uint64_t mac_count() const;
  Cycle last_completion_cycle() const;
  const PEValue& accumulator() const;

  PEValue mac(const PEValue& activation, const PEValue& weight);
  PEValue pass_activation(const PEValue& value) const;
  PEValue pass_weight(const PEValue& value) const;
  PEValue pass_result(const PEValue& value) const;

 private:
  void advance_cycles(Cycle cycles);
  void validate_config() const;
  void validate_operand(const PEValue& value,
                        const NumericFormatConfig& expected_format,
                        const char* operand_name) const;

  ProcessingElementConfig config_;
  PEValue accumulator_;
  std::uint64_t mac_count_{0};
  Cycle last_completion_cycle_{0};
};

bool same_numeric_format(const NumericFormatConfig& lhs,
                         const NumericFormatConfig& rhs);

}  // namespace npu_sim
