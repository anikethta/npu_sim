#include "npu_sim/backend/systemc/systemc_backend.hpp"

#include <atomic>
#include <cctype>
#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace npu_sim {

namespace {

std::uint64_t next_backend_instance_id() {
  static std::atomic<std::uint64_t> next_id{0};
  return next_id++;
}

std::string sanitize_module_name(std::string name) {
  if (name.empty()) {
    return "module";
  }

  for (char& character : name) {
    if (!std::isalnum(static_cast<unsigned char>(character)) &&
        character != '_') {
      character = '_';
    }
  }

  if (std::isdigit(static_cast<unsigned char>(name.front()))) {
    name.insert(name.begin(), '_');
  }

  return name;
}

PEValue make_pe_value(const NumericFormatConfig& format,
                      const PEInputValue& input,
                      const char* operand_name) {
  if (format.kind == NumericFormatKind::Float) {
    const double value =
        input.is_float ? input.floating_value
                       : static_cast<double>(input.integer_value);
    return PEValue::floating(format, value);
  }

  if (input.is_float) {
    throw std::invalid_argument(std::string("integer PE ") + operand_name + " input must be a Python int");
  }
  return PEValue::integer(format, input.integer_value);
}

PEMacResult make_mac_result(const PEValue& value, Cycle current_cycle,
                            std::uint64_t mac_count) {
  PEMacResult result;
  result.format = value.format();
  result.unknown = value.is_unknown();
  result.current_cycle = current_cycle;
  result.mac_count = mac_count;

  if (!result.unknown) {
    if (value.is_float()) {
      result.floating_value = value.floating_value();
    } else {
      result.integer_value = value.integer_value();
    }
  }

  return result;
}

}  // namespace

SystemCBackend::SystemCBackend(SystemCBackendConfig config)
    : config_(std::move(config)), instance_id_(next_backend_instance_id()) {
  scratchpads_.reserve(config_.scratchpads.size());
  for (std::size_t index = 0; index < config_.scratchpads.size(); ++index) {
    const SRAMScratchpadConfig& scratchpad_config = config_.scratchpads[index];
    const std::string module_name =
        "backend_" + std::to_string(instance_id_) + "_sram_" +
        std::to_string(index) + "_" +
        sanitize_module_name(scratchpad_config.name);
    scratchpads_.push_back(std::make_unique<SRAMScratchpad>(
        module_name.c_str(), scratchpad_config));
  }

  processing_elements_.reserve(config_.processing_elements.size());
  for (std::size_t index = 0; index < config_.processing_elements.size();
       ++index) {
    const ProcessingElementConfig& pe_config = config_.processing_elements[index];
    const std::string module_name =
        "backend_" + std::to_string(instance_id_) + "_pe_" +
        std::to_string(index) + "_" + sanitize_module_name(pe_config.name);
    auto pe = std::make_unique<ProcessingElement>(module_name.c_str(), pe_config);
    ProcessingElement* pe_ptr = pe.get();
    if (!processing_elements_by_name_.emplace(pe_config.name, pe_ptr).second) {
      throw std::invalid_argument("duplicate processing element name: " +
                                  pe_config.name);
    }
    processing_elements_.push_back(std::move(pe));
  }

  vector_processing_units_.reserve(config_.vector_processing_units.size());
  for (std::size_t index = 0; index < config_.vector_processing_units.size();
       ++index) {
    const VectorProcessingUnitConfig& vpu_config =
        config_.vector_processing_units[index];
    const std::string module_name =
        "backend_" + std::to_string(instance_id_) + "_vpu_" +
        std::to_string(index) + "_" + sanitize_module_name(vpu_config.name);
    auto vpu =
        std::make_unique<VectorProcessingUnit>(module_name.c_str(), vpu_config);
    VectorProcessingUnit* vpu_ptr = vpu.get();
    if (!vector_processing_units_by_name_.emplace(vpu_config.name, vpu_ptr)
             .second) {
      throw std::invalid_argument("duplicate VPU name: " + vpu_config.name);
    }
    vector_processing_units_.push_back(std::move(vpu));
  }

  systolic_arrays_.reserve(config_.systolic_arrays.size());
  for (std::size_t index = 0; index < config_.systolic_arrays.size();
       ++index) {
    const SystolicArrayConfig& array_config = config_.systolic_arrays[index];
    const std::string module_name =
        "backend_" + std::to_string(instance_id_) + "_array_" +
        std::to_string(index) + "_" + sanitize_module_name(array_config.name);
    auto array = std::make_unique<SystolicArray>(module_name.c_str(), array_config);
    SystolicArray* array_ptr = array.get();
    if (!systolic_arrays_by_name_.emplace(array_config.name, array_ptr)
             .second) {
      throw std::invalid_argument("duplicate systolic array name: " +
                                  array_config.name);
    }
    if (!array_config.output_vpu.empty()) {
      const auto vpu_found =
          vector_processing_units_by_name_.find(array_config.output_vpu);
      if (vpu_found == vector_processing_units_by_name_.end()) {
        throw std::out_of_range("unknown output VPU: " +
                                array_config.output_vpu);
      }
      array_ptr->attach_output_vpu(vpu_found->second);
    }
    systolic_arrays_.push_back(std::move(array));
  }
}

void SystemCBackend::run_until(Cycle max_cycle) {
  if (max_cycle > current_cycle_) {
    const Cycle delta = max_cycle - current_cycle_;
    sc_core::sc_start(cycle_time_ * static_cast<double>(delta));
    current_cycle_ = max_cycle;
  }
}

PEMacResult SystemCBackend::mac(const std::string& pe_name,
                                const PEInputValue& activation,
                                const PEInputValue& weight) {
  const auto found = processing_elements_by_name_.find(pe_name);
  if (found == processing_elements_by_name_.end()) {
    throw std::out_of_range("unknown processing element: " + pe_name);
  }

  ProcessingElement& pe = *found->second;
  const ProcessingElementConfig& pe_config = pe.config();
  const PEValue activation_value = make_pe_value(pe_config.activation.format, activation, "activation");
  const PEValue weight_value = make_pe_value(pe_config.weight.format, weight, "weight");
  const PEValue result = pe.mac(activation_value, weight_value);

  current_cycle_ = pe.last_completion_cycle();
  ++event_count_;
  return make_mac_result(result, current_cycle_, pe.mac_count());
}

SystolicArrayResult SystemCBackend::run_systolic_array(
    const std::string& array_name, const PEInputMatrix& activations,
    const PEInputMatrix& weights, bool trace_enabled) {
  return run_systolic_array_stream(array_name, {activations}, {weights},
                                   trace_enabled);
}

SystolicArrayResult SystemCBackend::run_systolic_array_stream(
    const std::string& array_name,
    const PEInputMatrixBatch& activation_batches,
    const PEInputMatrixBatch& weight_batches, bool trace_enabled) {
  const auto found = systolic_arrays_by_name_.find(array_name);
  if (found == systolic_arrays_by_name_.end()) {
    throw std::out_of_range("unknown systolic array: " + array_name);
  }

  SystolicArrayResult result =
      found->second->run_matrix_multiply_stream(
          activation_batches, weight_batches, trace_enabled);
  current_cycle_ = result.current_cycle;
  ++event_count_;
  return result;
}

Cycle SystemCBackend::current_cycle() const { return current_cycle_; }

std::uint64_t SystemCBackend::component_count() const {
  std::uint64_t array_component_count = 0;
  for (const auto& array : systolic_arrays_) {
    array_component_count += array->component_count();
  }
  std::uint64_t vpu_component_count = 0;
  for (const auto& vpu : vector_processing_units_) {
    vpu_component_count += vpu->component_count();
  }
  return config_.core_count + scratchpads_.size() + processing_elements_.size() +
         vpu_component_count + array_component_count;
}

std::uint64_t SystemCBackend::event_count() const { return event_count_; }

}  // namespace npu_sim
