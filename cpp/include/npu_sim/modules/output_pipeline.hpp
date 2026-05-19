#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include <systemc>

#include "npu_sim/modules/processing_element.hpp"

namespace npu_sim {

struct PipelineRegister {
  bool per_column{false};
  std::vector<double> values{1.0};
};

using RequantizationRegister = PipelineRegister;

struct BiasAdderConfig {
  bool enabled{false};
  NumericFormatConfig format{NumericFormatKind::SignedInt, 16, 0, 0};
  PipelineRegister bias{false, {0.0}};
  std::string placement{"after_systolic_array"};
};

struct RequantizationConfig {
  bool enabled{false};
  NumericFormatConfig target;
  PipelineRegister scale_multiplier{false, {1.0}};
  PipelineRegister shift{false, {0.0}};
  bool rounding{true};
  std::string placement{"after_systolic_array"};
};

struct OutputPipelineStage {
  std::string event;
  PEValue value;
  double scale_multiplier{1.0};
  std::int64_t shift{0};
  std::string target_kind;
  std::uint64_t target_bits{0};
};

class BiasAdderPipeline : public sc_core::sc_module {
 public:
  BiasAdderPipeline(sc_core::sc_module_name module_name,
                    BiasAdderConfig config);

  const BiasAdderConfig& config() const;
  Cycle latency() const;
  PEValue apply(const PEValue& accumulator, std::uint64_t col) const;
  OutputPipelineStage trace_stage(const PEValue& accumulator,
                                  std::uint64_t col) const;

 private:
  BiasAdderConfig config_;
};

class RequantizationPipeline : public sc_core::sc_module {
 public:
  RequantizationPipeline(sc_core::sc_module_name module_name,
                         RequantizationConfig config);

  const RequantizationConfig& config() const;
  Cycle latency() const;
  PEValue apply(const PEValue& value, std::uint64_t col) const;
  std::vector<OutputPipelineStage> trace_stages(const PEValue& value,
                                                std::uint64_t col) const;

 private:
  RequantizationConfig config_;
};

double register_value(const PipelineRegister& reg, std::uint64_t col,
                      const char* name);
std::int64_t integer_register_value(const PipelineRegister& reg,
                                    std::uint64_t col, const char* name);

}  // namespace npu_sim
