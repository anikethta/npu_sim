#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <systemc>

#include "npu_sim/modules/output_pipeline.hpp"
#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/modules/vector_processing_unit.hpp"
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
  BiasAdderConfig bias;
  struct SPUOutputFIFOConfig {
    std::uint64_t depth{16};
    Cycle latency_cycles{0};
  } spu_output_fifo;
  std::string output_vpu;
  RequantizationConfig requantization;
};

struct SPUOutputFIFOLane {
  bool valid{false};
  std::int64_t row{-1};
  std::int64_t col{-1};
  double value{0.0};
  bool value_is_float{false};
};

struct SPUOutputFIFOEntry {
  std::vector<SPUOutputFIFOLane> lanes;
};

struct SystolicTraceEvent {
  Cycle cycle{0};
  std::string event;
  std::string array;
  std::uint64_t batch{0};
  std::int64_t row{-1};
  std::int64_t col{-1};
  std::int64_t k{-1};
  double value{0.0};
  bool value_is_float{false};
  bool has_value{false};
  bool has_requantization_metadata{false};
  double scale_multiplier{1.0};
  std::int64_t shift{0};
  bool has_bias_metadata{false};
  double bias{0.0};
  std::string bias_format_kind;
  std::uint64_t bias_format_bits{0};
  std::string placement;
  std::string target_kind;
  std::uint64_t target_bits{0};
  bool has_spu_fifo_metadata{false};
  bool dequeue_asserted{false};
  std::vector<SPUOutputFIFOLane> lanes;
};

struct SystolicArrayResult {
  std::vector<std::vector<PEMacResult>> outputs;
  std::vector<std::vector<std::vector<PEMacResult>>> output_batches;
  std::vector<SystolicTraceEvent> trace;
  Cycle current_cycle{0};
  std::uint64_t operation_count{0};
  PEDataflowMode dataflow_mode{PEDataflowMode::OutputStationary};
  std::uint64_t size{0};
};

using PEInputMatrix = std::vector<std::vector<PEInputValue>>;
using PEInputMatrixBatch = std::vector<PEInputMatrix>;

class SystolicArray : public sc_core::sc_module {
 public:
  SystolicArray(sc_core::sc_module_name module_name,
                SystolicArrayConfig config);

  const std::string& name() const;
  const SystolicArrayConfig& config() const;
  std::uint64_t size() const;
  std::uint64_t component_count() const;
  void attach_output_vpu(VectorProcessingUnit* vpu);

  SystolicArrayResult run_matrix_multiply(const PEInputMatrix& activations,
                                          const PEInputMatrix& weights,
                                          bool trace_enabled);
  SystolicArrayResult run_matrix_multiply_stream(
      const PEInputMatrixBatch& activation_batches,
      const PEInputMatrixBatch& weight_batches, bool trace_enabled);

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
  Cycle output_pipeline_latency() const;
  bool has_output_pipeline() const;
  bool uses_vpu() const;
  bool has_after_systolic_requantization() const;
  bool has_after_vpu_requantization() const;
  bool uses_inside_pe_bias() const;
  PEValue bias_value_for_column(std::uint64_t col) const;
  PEValue process_output(const PEValue& accumulator, std::uint64_t col) const;
  struct OutputTraceItem {
    OutputTraceItem(Cycle cycle, std::uint64_t batch, std::int64_t row,
                    std::int64_t col, PEValue accumulator)
        : cycle(cycle),
          batch(batch),
          row(row),
          col(col),
          accumulator(std::move(accumulator)) {}

    Cycle cycle;
    std::uint64_t batch;
    std::int64_t row;
    std::int64_t col;
    PEValue accumulator;
  };
  void add_output_pipeline_traces(
      std::vector<SystolicTraceEvent>& trace,
      const std::vector<OutputTraceItem>& items) const;
  void add_trace_value(std::vector<SystolicTraceEvent>& trace, Cycle cycle,
                       std::string event, std::uint64_t batch,
                       std::int64_t row, std::int64_t col, std::int64_t k,
                       const PEValue& value) const;
  void add_bias_trace_value(std::vector<SystolicTraceEvent>& trace,
                            Cycle cycle, std::uint64_t batch,
                            std::int64_t row, std::int64_t col,
                            const PEValue& value) const;
  void add_inside_pe_bias_trace_value(std::vector<SystolicTraceEvent>& trace,
                                      Cycle cycle, std::string event,
                                      std::uint64_t batch, std::int64_t row,
                                      std::int64_t col,
                                      const PEValue& value) const;
  void add_inside_pe_bias_stream_trace(std::vector<SystolicTraceEvent>& trace,
                                       Cycle base_cycle, std::uint64_t batch,
                                       const std::vector<std::vector<PEValue>>&
                                           accumulators) const;
  void add_requant_trace_value(std::vector<SystolicTraceEvent>& trace,
                               Cycle cycle, const OutputPipelineStage& stage,
                               std::uint64_t batch, std::int64_t row,
                               std::int64_t col,
                               const std::string& placement) const;
  void add_trace_no_value(std::vector<SystolicTraceEvent>& trace, Cycle cycle,
                          std::string event, std::uint64_t batch,
                          std::int64_t row, std::int64_t col,
                          std::int64_t k) const;
  void add_spu_fifo_trace_value(std::vector<SystolicTraceEvent>& trace,
                                Cycle cycle, std::string event,
                                std::uint64_t batch,
                                bool dequeue_asserted,
                                const SPUOutputFIFOEntry& entry) const;
  Cycle current_cycle_from_systemc() const;
  std::vector<std::vector<PEValue>> convert_matrix(
      const PEInputMatrix& matrix, const NumericFormatConfig& format,
      const char* operand_name) const;
  std::vector<std::vector<PEValue>> initial_accumulators() const;
  std::vector<std::vector<PEMacResult>> make_output_matrix(
      const std::vector<std::vector<PEValue>>& accumulators,
      Cycle current_cycle) const;
  Cycle output_stationary_schedule(
      const std::vector<std::vector<PEValue>>& activations,
      const std::vector<std::vector<PEValue>>& weights,
      std::vector<std::vector<PEValue>>& accumulators,
      std::vector<SystolicTraceEvent>& trace, bool trace_enabled,
      Cycle base_cycle, std::uint64_t batch) const;
  Cycle weight_stationary_schedule(
      const std::vector<std::vector<PEValue>>& activations,
      const std::vector<std::vector<PEValue>>& weights,
      std::vector<std::vector<PEValue>>& accumulators,
      std::vector<SystolicTraceEvent>& trace, bool trace_enabled,
      Cycle load_base_cycle, Cycle compute_base_cycle,
      std::uint64_t batch, bool load_weights, bool use_shadow_weights) const;

  SystolicArrayConfig config_;
  std::vector<std::unique_ptr<ProcessingElement>> pes_;
  mutable std::unique_ptr<sc_core::sc_fifo<std::shared_ptr<SPUOutputFIFOEntry>>>
      spu_output_fifo_;
  std::unique_ptr<BiasAdderPipeline> bias_pipeline_;
  std::unique_ptr<RequantizationPipeline> after_systolic_requantization_;
  VectorProcessingUnit* vpu_{nullptr};
  std::unique_ptr<RequantizationPipeline> after_vpu_requantization_;
  Cycle last_completion_cycle_{0};
  std::uint64_t operation_count_{0};
};

}  // namespace npu_sim
