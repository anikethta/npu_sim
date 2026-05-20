#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <systemc>

#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/modules/sram_scratchpad.hpp"
#include "npu_sim/types.hpp"

namespace npu_sim {

struct VectorProcessingUnitConfig {
  std::string name;
  bool enabled{false};
  std::uint64_t lanes{1};
  Cycle latency_cycles{1};
  SRAMScratchpadConfig instruction_memory{"vpu_instruction_sram", 1, 0, 0,
                                          32, 32, 64, 1, 1};
};

class VectorProcessingUnit : public sc_core::sc_module {
 public:
  VectorProcessingUnit(sc_core::sc_module_name module_name,
                       VectorProcessingUnitConfig config);

  const std::string& name() const;
  const VectorProcessingUnitConfig& config() const;
  Cycle latency() const;
  std::uint64_t component_count() const;
  PEValue process(const PEValue& value) const;

 private:
  VectorProcessingUnitConfig config_;
  std::unique_ptr<SRAMScratchpad> instruction_memory_;
};

}  // namespace npu_sim
