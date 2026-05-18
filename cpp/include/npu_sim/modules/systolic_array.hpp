#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <systemc>

#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/types.hpp"

namespace npu_sim {

struct SystolicArrayConfig {
  std::string name;
  std::uint64_t size{1};
  Cycle mac_latency_cycles{1};
  PEDataflowMode dataflow_mode{PEDataflowMode::OutputStationary};
  PEOperandConfig activation;
  PEOperandConfig weight;
  NumericFormatConfig accumulator;
};

struct SystolicTraceEvent {
  Cycle cycle{0};
  std::string event;
  std::string array;
  std::int64_t row{-1};
  std::int64_t col{-1};
  std::int64_t k{-1};
  double value{0.0};
  bool value_is_float{false};
  bool has_value{false};
};

struct SystolicArrayResult {
  std::vector<std::vector<PEMacResult>> outputs;
  std::vector<SystolicTraceEvent> trace;
  Cycle current_cycle{0};
  std::uint64_t operation_count{0};
  PEDataflowMode dataflow_mode{PEDataflowMode::OutputStationary};
  std::uint64_t size{0};
};

using PEInputMatrix = std::vector<std::vector<PEInputValue>>;

class SystolicArray : public sc_core::sc_module {
 public:
  SystolicArray(sc_core::sc_module_name module_name,
                SystolicArrayConfig config);

  const std::string& name() const;
  const SystolicArrayConfig& config() const;
  std::uint64_t size() const;
  std::uint64_t component_count() const;

  SystolicArrayResult run_matrix_multiply(const PEInputMatrix& activations,
                                          const PEInputMatrix& weights,
                                          bool trace_enabled);

 private:
  void validate_config() const;
  void validate_matrix_shape(const PEInputMatrix& matrix,
                             const char* matrix_name) const;
  PEValue make_value(const NumericFormatConfig& format,
                     const PEInputValue& input,
                     const char* operand_name) const;
  PEValue multiply_accumulate(const PEValue& accumulator,
                              const PEValue& activation,
                              const PEValue& weight) const;
  PEMacResult make_result(const PEValue& value, Cycle current_cycle,
                          std::uint64_t mac_count) const;
  void add_trace_value(std::vector<SystolicTraceEvent>& trace, Cycle cycle,
                       std::string event, std::int64_t row, std::int64_t col,
                       std::int64_t k, const PEValue& value) const;
  void add_trace_no_value(std::vector<SystolicTraceEvent>& trace, Cycle cycle,
                          std::string event, std::int64_t row,
                          std::int64_t col, std::int64_t k) const;
  Cycle current_cycle_from_systemc() const;
  Cycle output_stationary_schedule(
      const std::vector<std::vector<PEValue>>& activations,
      const std::vector<std::vector<PEValue>>& weights,
      std::vector<std::vector<PEValue>>& accumulators,
      std::vector<SystolicTraceEvent>& trace, bool trace_enabled) const;
  Cycle weight_stationary_schedule(
      const std::vector<std::vector<PEValue>>& activations,
      const std::vector<std::vector<PEValue>>& weights,
      std::vector<std::vector<PEValue>>& accumulators,
      std::vector<SystolicTraceEvent>& trace, bool trace_enabled) const;

  SystolicArrayConfig config_;
  std::vector<std::unique_ptr<ProcessingElement>> pes_;
  Cycle last_completion_cycle_{0};
  std::uint64_t operation_count_{0};
};

}  // namespace npu_sim
