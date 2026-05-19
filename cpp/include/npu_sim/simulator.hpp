#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/modules/sram_scratchpad.hpp"
#include "npu_sim/modules/systolic_array.hpp"
#include "npu_sim/types.hpp"

namespace npu_sim {

class SystemCBackend;

struct SimulatorConfig {
  std::uint64_t core_count{1};
  std::vector<SRAMScratchpadConfig> scratchpads;
  std::vector<ProcessingElementConfig> processing_elements;
  std::vector<SystolicArrayConfig> systolic_arrays;
};

struct SimulatorStats {
  Cycle current_cycle{0};
  std::uint64_t event_count{0};
  std::uint64_t component_count{0};
};

class Simulator {
 public:
  explicit Simulator(SimulatorConfig config);
  ~Simulator();

  Simulator(const Simulator&) = delete;
  Simulator& operator=(const Simulator&) = delete;
  Simulator(Simulator&&) noexcept;
  Simulator& operator=(Simulator&&) noexcept;

  void schedule_event(Cycle delay, std::string name);
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
  SimulatorStats run();
  SimulatorStats run_until(Cycle max_cycle);
  SimulatorStats stats() const;

 private:
  SimulatorConfig config_;
  std::unique_ptr<SystemCBackend> backend_;
};

}  // namespace npu_sim
