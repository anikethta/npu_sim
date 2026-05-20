#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <systemc>

#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/modules/sram_scratchpad.hpp"
#include "npu_sim/modules/systolic_array.hpp"
#include "npu_sim/modules/vector_processing_unit.hpp"
#include "npu_sim/types.hpp"

namespace npu_sim {

struct SystemCBackendConfig {
  std::uint64_t core_count{1};
  std::vector<SRAMScratchpadConfig> scratchpads;
  std::vector<ProcessingElementConfig> processing_elements;
  std::vector<VectorProcessingUnitConfig> vector_processing_units;
  std::vector<SystolicArrayConfig> systolic_arrays;
};

class SystemCBackend {
 public:
  explicit SystemCBackend(SystemCBackendConfig config);

  void run_until(Cycle max_cycle);
  PEMacResult mac(const std::string& pe_name, const PEInputValue& activation,
                  const PEInputValue& weight);
  SystolicArrayResult run_systolic_array(const std::string& array_name,
                                         const PEInputMatrix& activations,
                                         const PEInputMatrix& weights,
                                         bool trace_enabled);
  SystolicArrayResult run_systolic_array_stream(
      const std::string& array_name,
      const PEInputMatrixBatch& activation_batches,
      const PEInputMatrixBatch& weight_batches, bool trace_enabled);
  Cycle current_cycle() const;
  std::uint64_t component_count() const;
  std::uint64_t event_count() const;

 private:
  SystemCBackendConfig config_;
  std::uint64_t instance_id_{0};
  std::vector<std::unique_ptr<SRAMScratchpad>> scratchpads_;
  std::vector<std::unique_ptr<ProcessingElement>> processing_elements_;
  std::unordered_map<std::string, ProcessingElement*>
      processing_elements_by_name_;
  std::vector<std::unique_ptr<VectorProcessingUnit>> vector_processing_units_;
  std::unordered_map<std::string, VectorProcessingUnit*>
      vector_processing_units_by_name_;
  std::vector<std::unique_ptr<SystolicArray>> systolic_arrays_;
  std::unordered_map<std::string, SystolicArray*> systolic_arrays_by_name_;
  sc_core::sc_time cycle_time_{1.0, sc_core::SC_NS};
  Cycle current_cycle_{0};
  std::uint64_t event_count_{0};
};

}  // namespace npu_sim
