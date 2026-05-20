#include "npu_sim/modules/vector_processing_unit.hpp"

#include <stdexcept>
#include <string>
#include <utility>

namespace npu_sim {

VectorProcessingUnit::VectorProcessingUnit(sc_core::sc_module_name module_name,
                                           VectorProcessingUnitConfig config)
    : sc_core::sc_module(module_name), config_(std::move(config)) {
  if (config_.name.empty()) {
    throw std::invalid_argument("VPU name must be non-empty");
  }
  if (config_.lanes == 0) {
    throw std::invalid_argument("VPU lanes must be positive");
  }
  const std::string instruction_memory_name = "instruction_sram";
  instruction_memory_ = std::make_unique<SRAMScratchpad>(
      instruction_memory_name.c_str(), config_.instruction_memory);
}

const std::string& VectorProcessingUnit::name() const { return config_.name; }

const VectorProcessingUnitConfig& VectorProcessingUnit::config() const {
  return config_;
}

Cycle VectorProcessingUnit::latency() const {
  return config_.enabled ? config_.latency_cycles : 0;
}

std::uint64_t VectorProcessingUnit::component_count() const {
  return (config_.enabled ? 1 : 0) + (instruction_memory_ ? 1 : 0);
}

PEValue VectorProcessingUnit::process(const PEValue& value) const {
  return value;
}

}  // namespace npu_sim
