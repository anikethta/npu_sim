#include "npu_sim/simulator.hpp"

#include <utility>

#include "npu_sim/backend/systemc/systemc_backend.hpp"

namespace npu_sim {

namespace {

SystemCBackendConfig to_backend_config(const SimulatorConfig& config) {
  SystemCBackendConfig backend_config;
  backend_config.core_count = config.core_count;
  backend_config.scratchpads = config.scratchpads;
  backend_config.processing_elements = config.processing_elements;
  backend_config.systolic_arrays = config.systolic_arrays;
  return backend_config;
}

}  // namespace

Simulator::Simulator(SimulatorConfig config)
    : config_(std::move(config)),
      backend_(std::make_unique<SystemCBackend>(to_backend_config(config_))) {}

Simulator::~Simulator() = default;

Simulator::Simulator(Simulator&&) noexcept = default;

Simulator& Simulator::operator=(Simulator&&) noexcept = default;

void Simulator::schedule_event(Cycle delay, std::string name) {
  (void)delay;
  (void)name;
}

PEMacResult Simulator::mac(const std::string& pe_name,
                           const PEInputValue& activation,
                           const PEInputValue& weight) {
  return backend_->mac(pe_name, activation, weight);
}

SystolicArrayResult Simulator::run_systolic_array(
    const std::string& array_name, const PEInputMatrix& activations,
    const PEInputMatrix& weights, bool trace_enabled) {
  return backend_->run_systolic_array(array_name, activations, weights,
                                      trace_enabled);
}

SystolicArrayResult Simulator::run_systolic_array_stream(
    const std::string& array_name,
    const PEInputMatrixBatch& activation_batches,
    const PEInputMatrixBatch& weight_batches, bool trace_enabled) {
  return backend_->run_systolic_array_stream(
      array_name, activation_batches, weight_batches, trace_enabled);
}

SimulatorStats Simulator::run() { return stats(); }

SimulatorStats Simulator::run_until(Cycle max_cycle) {
  backend_->run_until(max_cycle);
  return stats();
}

SimulatorStats Simulator::stats() const {
  return SimulatorStats{
      backend_->current_cycle(),
      backend_->event_count(),
      backend_->component_count(),
  };
}

}  // namespace npu_sim
