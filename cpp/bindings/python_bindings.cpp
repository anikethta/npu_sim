#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cmath>
#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>

#include "npu_sim/simulator.hpp"

namespace py = pybind11;

namespace {

npu_sim::NumericFormatKind parse_numeric_format_kind(const std::string& kind) {
  if (kind == "signed_int") {
    return npu_sim::NumericFormatKind::SignedInt;
  }
  if (kind == "unsigned_int") {
    return npu_sim::NumericFormatKind::UnsignedInt;
  }
  if (kind == "float") {
    return npu_sim::NumericFormatKind::Float;
  }
  throw std::invalid_argument("unknown numeric format kind: " + kind);
}

npu_sim::PEDataflowMode parse_pe_dataflow_mode(const std::string& mode) {
  if (mode == "output_stationary") {
    return npu_sim::PEDataflowMode::OutputStationary;
  }
  if (mode == "weight_stationary") {
    return npu_sim::PEDataflowMode::WeightStationary;
  }
  throw std::invalid_argument("unknown PE dataflow mode: " + mode);
}

npu_sim::NumericFormatConfig parse_numeric_format(const py::dict& format) {
  npu_sim::NumericFormatConfig native_format;
  native_format.kind =
      parse_numeric_format_kind(format["kind"].cast<std::string>());
  native_format.bits = format["bits"].cast<std::uint64_t>();
  native_format.exponent_bits =
      format["exponent_bits"].cast<std::uint64_t>();
  native_format.mantissa_bits =
      format["mantissa_bits"].cast<std::uint64_t>();
  return native_format;
}

npu_sim::PEOperandConfig parse_pe_operand(const py::dict& operand) {
  npu_sim::PEOperandConfig native_operand;
  native_operand.format = parse_numeric_format(operand["format"].cast<py::dict>());
  return native_operand;
}

npu_sim::RequantizationRegister parse_requantization_register(
    const py::dict& reg) {
  npu_sim::RequantizationRegister native_reg;
  native_reg.per_column = reg["per_column"].cast<bool>();
  native_reg.values = reg["values"].cast<std::vector<double>>();
  return native_reg;
}

npu_sim::BiasAdderConfig parse_bias(const py::dict& bias) {
  npu_sim::BiasAdderConfig native_bias;
  native_bias.enabled = bias["enabled"].cast<bool>();
  native_bias.format = parse_numeric_format(bias["format"].cast<py::dict>());
  native_bias.bias =
      parse_requantization_register(bias["bias"].cast<py::dict>());
  native_bias.placement = bias["placement"].cast<std::string>();
  return native_bias;
}

npu_sim::RequantizationConfig parse_requantization(
    const py::dict& requantization) {
  npu_sim::RequantizationConfig native_requantization;
  native_requantization.enabled = requantization["enabled"].cast<bool>();
  native_requantization.target =
      parse_numeric_format(requantization["target"].cast<py::dict>());
  native_requantization.scale_multiplier = parse_requantization_register(
      requantization["scale_multiplier"].cast<py::dict>());
  native_requantization.shift =
      parse_requantization_register(requantization["shift"].cast<py::dict>());
  native_requantization.rounding = requantization["rounding"].cast<bool>();
  native_requantization.placement = requantization["placement"].cast<std::string>();
  return native_requantization;
}

npu_sim::SystolicArrayConfig::SPUOutputFIFOConfig parse_spu_output_fifo(
    const py::dict& fifo) {
  npu_sim::SystolicArrayConfig::SPUOutputFIFOConfig native_fifo;
  native_fifo.depth = fifo["depth"].cast<std::uint64_t>();
  native_fifo.latency_cycles =
      fifo["latency_cycles"].cast<npu_sim::Cycle>();
  return native_fifo;
}

npu_sim::SRAMScratchpadConfig parse_scratchpad_config(
    const py::dict& scratchpad) {
  npu_sim::SRAMScratchpadConfig scratchpad_config;
  scratchpad_config.name = scratchpad["name"].cast<std::string>();
  scratchpad_config.num_rw_ports =
      scratchpad["num_rw_ports"].cast<std::uint64_t>();
  scratchpad_config.num_r_ports =
      scratchpad["num_r_ports"].cast<std::uint64_t>();
  scratchpad_config.num_w_ports =
      scratchpad["num_w_ports"].cast<std::uint64_t>();
  scratchpad_config.word_size =
      scratchpad["word_size"].cast<std::uint64_t>();
  scratchpad_config.write_size =
      scratchpad["write_size"].cast<std::uint64_t>();
  scratchpad_config.num_words =
      scratchpad["num_words"].cast<std::uint64_t>();
  scratchpad_config.read_latency_cycles =
      scratchpad["read_latency_cycles"].cast<npu_sim::Cycle>();
  scratchpad_config.write_latency_cycles =
      scratchpad["write_latency_cycles"].cast<npu_sim::Cycle>();
  return scratchpad_config;
}

npu_sim::VectorProcessingUnitConfig parse_vpu(const py::dict& vpu) {
  npu_sim::VectorProcessingUnitConfig native_vpu;
  native_vpu.name = vpu["name"].cast<std::string>();
  native_vpu.enabled = vpu["enabled"].cast<bool>();
  native_vpu.lanes = vpu["lanes"].cast<std::uint64_t>();
  native_vpu.latency_cycles = vpu["latency_cycles"].cast<npu_sim::Cycle>();
  native_vpu.instruction_memory =
      parse_scratchpad_config(vpu["instruction_memory"].cast<py::dict>());
  return native_vpu;
}

npu_sim::SystolicArrayConfig parse_systolic_array(const py::dict& array) {
  npu_sim::SystolicArrayConfig array_config;
  array_config.name = array["name"].cast<std::string>();
  array_config.size = array["size"].cast<std::uint64_t>();
  array_config.mac_latency_cycles =
      array["mac_latency_cycles"].cast<npu_sim::Cycle>();
  array_config.dataflow_mode =
      parse_pe_dataflow_mode(array["dataflow_mode"].cast<std::string>());
  array_config.activation =
      parse_pe_operand(array["activation"].cast<py::dict>());
  array_config.weight = parse_pe_operand(array["weight"].cast<py::dict>());
  array_config.accumulator =
      parse_numeric_format(array["accumulator"].cast<py::dict>());
  array_config.bias = parse_bias(array["bias"].cast<py::dict>());
  if (array.contains("spu_output_fifo")) {
    array_config.spu_output_fifo =
        parse_spu_output_fifo(array["spu_output_fifo"].cast<py::dict>());
  }
  if (array.contains("vpu")) {
    throw std::invalid_argument(
        "SystolicArrayConfig no longer accepts nested vpu; use "
        "NPUConfig.vector_processing_units and output_vpu");
  }
  if (!array["output_vpu"].is_none()) {
    array_config.output_vpu = array["output_vpu"].cast<std::string>();
  }
  array_config.requantization =
      parse_requantization(array["requantization"].cast<py::dict>());
  return array_config;
}

npu_sim::SimulatorConfig parse_config(const py::dict& config) {
  npu_sim::SimulatorConfig native_config;

  if (config.contains("cores")) {
    const py::dict cores = config["cores"].cast<py::dict>();
    if (cores.contains("count")) {
      native_config.core_count = cores["count"].cast<std::uint64_t>();
    }
  }

  if (native_config.core_count == 0) {
    throw std::invalid_argument("core count must be positive");
  }

  if (config.contains("scratchpads")) {
    for (const py::handle item : config["scratchpads"]) {
      native_config.scratchpads.push_back(
          parse_scratchpad_config(item.cast<py::dict>()));
    }
  }

  if (config.contains("processing_elements")) {
    for (const py::handle item : config["processing_elements"]) {
      const py::dict pe = item.cast<py::dict>();
      npu_sim::ProcessingElementConfig pe_config;
      pe_config.name = pe["name"].cast<std::string>();
      pe_config.mac_latency_cycles =
          pe["mac_latency_cycles"].cast<npu_sim::Cycle>();
      pe_config.dataflow_mode =
          parse_pe_dataflow_mode(pe["dataflow_mode"].cast<std::string>());
      pe_config.activation = parse_pe_operand(pe["activation"].cast<py::dict>());
      pe_config.weight = parse_pe_operand(pe["weight"].cast<py::dict>());
      pe_config.accumulator =
          parse_numeric_format(pe["accumulator"].cast<py::dict>());
      native_config.processing_elements.push_back(std::move(pe_config));
    }
  }

  if (config.contains("vector_processing_units")) {
    for (const py::handle item : config["vector_processing_units"]) {
      native_config.vector_processing_units.push_back(
          parse_vpu(item.cast<py::dict>()));
    }
  }

  if (config.contains("systolic_arrays")) {
    for (const py::handle item : config["systolic_arrays"]) {
      native_config.systolic_arrays.push_back(
          parse_systolic_array(item.cast<py::dict>()));
    }
  }

  return native_config;
}

py::dict stats_to_dict(const npu_sim::SimulatorStats& stats) {
  py::dict out;
  out["current_cycle"] = stats.current_cycle;
  out["event_count"] = stats.event_count;
  out["component_count"] = stats.component_count;
  return out;
}

npu_sim::PEInputValue parse_pe_input(const py::object& value) {
  npu_sim::PEInputValue input;
  if (py::isinstance<py::bool_>(value)) {
    throw py::type_error("PE inputs must be int or float, not bool");
  }
  if (py::isinstance<py::int_>(value)) {
    input.is_float = false;
    input.integer_value = value.cast<std::int64_t>();
    input.floating_value = static_cast<double>(input.integer_value);
    return input;
  }
  if (py::isinstance<py::float_>(value)) {
    input.is_float = true;
    input.floating_value = value.cast<double>();
    return input;
  }
  throw py::type_error("PE inputs must be int or float");
}

npu_sim::PEInputMatrix parse_pe_input_matrix(const py::object& matrix) {
  npu_sim::PEInputMatrix native_matrix;
  const py::sequence rows = matrix.cast<py::sequence>();
  native_matrix.reserve(static_cast<std::size_t>(py::len(rows)));
  for (const py::handle row_handle : rows) {
    const py::sequence row = row_handle.cast<py::sequence>();
    std::vector<npu_sim::PEInputValue> native_row;
    native_row.reserve(static_cast<std::size_t>(py::len(row)));
    for (const py::handle value : row) {
      native_row.push_back(parse_pe_input(py::reinterpret_borrow<py::object>(
          value)));
    }
    native_matrix.push_back(std::move(native_row));
  }
  return native_matrix;
}

npu_sim::PEInputMatrixBatch parse_pe_input_matrix_batch(
    const py::object& matrices) {
  npu_sim::PEInputMatrixBatch native_matrices;
  const py::sequence batch = matrices.cast<py::sequence>();
  native_matrices.reserve(static_cast<std::size_t>(py::len(batch)));
  for (const py::handle matrix : batch) {
    native_matrices.push_back(parse_pe_input_matrix(
        py::reinterpret_borrow<py::object>(matrix)));
  }
  return native_matrices;
}

std::string numeric_format_kind_to_string(npu_sim::NumericFormatKind kind) {
  switch (kind) {
    case npu_sim::NumericFormatKind::SignedInt:
      return "signed_int";
    case npu_sim::NumericFormatKind::UnsignedInt:
      return "unsigned_int";
    case npu_sim::NumericFormatKind::Float:
      return "float";
  }
  throw std::invalid_argument("unknown numeric format kind");
}

py::dict mac_result_to_dict(const npu_sim::PEMacResult& result) {
  py::dict out;
  out["kind"] = numeric_format_kind_to_string(result.format.kind);
  out["bits"] = result.format.bits;
  out["exponent_bits"] = result.format.exponent_bits;
  out["mantissa_bits"] = result.format.mantissa_bits;
  out["unknown"] = result.unknown;
  out["current_cycle"] = result.current_cycle;
  out["mac_count"] = result.mac_count;

  if (result.unknown) {
    out["value"] = py::none();
  } else if (result.format.kind == npu_sim::NumericFormatKind::Float) {
    out["value"] = result.floating_value;
  } else {
    out["value"] = result.integer_value;
  }

  return out;
}

py::dict trace_event_to_dict(const npu_sim::SystolicTraceEvent& event) {
  py::dict out;
  out["cycle"] = event.cycle;
  out["event"] = event.event;
  out["array"] = event.array;
  out["batch"] = event.batch;
  out["row"] = event.row < 0 ? py::none() : py::cast(event.row);
  out["col"] = event.col < 0 ? py::none() : py::cast(event.col);
  out["k"] = event.k < 0 ? py::none() : py::cast(event.k);
  if (event.has_value) {
    if (event.value_is_float) {
      out["value"] = event.value;
    } else {
      out["value"] = static_cast<std::int64_t>(std::llround(event.value));
    }
  } else {
    out["value"] = py::none();
  }
  if (event.has_requantization_metadata) {
    py::dict metadata;
    metadata["scale_multiplier"] = event.scale_multiplier;
    metadata["shift"] = event.shift;
    metadata["placement"] = event.placement;
    metadata["target_kind"] = event.target_kind;
    metadata["target_bits"] = event.target_bits;
    out["requantization"] = std::move(metadata);
  } else {
    out["requantization"] = py::none();
  }
  if (event.has_bias_metadata) {
    py::dict metadata;
    metadata["bias"] = event.bias;
    metadata["format_kind"] = event.bias_format_kind;
    metadata["format_bits"] = event.bias_format_bits;
    metadata["placement"] = event.placement;
    out["bias"] = std::move(metadata);
  } else {
    out["bias"] = py::none();
  }
  if (event.has_spu_fifo_metadata) {
    py::list lanes;
    for (const npu_sim::SPUOutputFIFOLane& lane : event.lanes) {
      py::dict lane_out;
      lane_out["valid"] = lane.valid;
      lane_out["row"] = lane.row < 0 ? py::none() : py::cast(lane.row);
      lane_out["col"] = lane.col < 0 ? py::none() : py::cast(lane.col);
      lane_out["value"] =
          lane.value_is_float
              ? py::cast(lane.value)
              : py::cast(static_cast<std::int64_t>(std::llround(lane.value)));
      lanes.append(std::move(lane_out));
    }
    out["lanes"] = std::move(lanes);
    out["dequeue_asserted"] = event.dequeue_asserted;
  } else {
    out["lanes"] = py::none();
    out["dequeue_asserted"] = py::none();
  }
  return out;
}

py::dict systolic_result_to_dict(const npu_sim::SystolicArrayResult& result) {
  py::dict out;
  auto output_matrix_to_list =
      [](const std::vector<std::vector<npu_sim::PEMacResult>>& matrix) {
        py::list outputs;
        for (const auto& row : matrix) {
          py::list output_row;
          for (const npu_sim::PEMacResult& value : row) {
            output_row.append(mac_result_to_dict(value)["value"]);
          }
          outputs.append(std::move(output_row));
        }
        return outputs;
      };

  py::list output_batches;
  for (const auto& matrix : result.output_batches) {
    output_batches.append(output_matrix_to_list(matrix));
  }
  py::list outputs = output_matrix_to_list(result.outputs);
  py::list trace;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    trace.append(trace_event_to_dict(event));
  }
  out["outputs"] = std::move(outputs);
  out["output_batches"] = std::move(output_batches);
  out["batch_count"] = result.output_batches.size();
  out["current_cycle"] = result.current_cycle;
  out["operation_count"] = result.operation_count;
  out["dataflow_mode"] =
      result.dataflow_mode == npu_sim::PEDataflowMode::WeightStationary
          ? "weight_stationary"
          : "output_stationary";
  out["size"] = result.size;
  out["trace"] = std::move(trace);
  return out;
}

}  // namespace

PYBIND11_MODULE(_native, m) {
  m.doc() = "Native backend for npu_sim";

  py::class_<npu_sim::Simulator>(m, "Simulator")
      .def(py::init([](const py::dict& config) {
        return npu_sim::Simulator(parse_config(config));
      }))
      .def("schedule_event", &npu_sim::Simulator::schedule_event,
           py::arg("delay"), py::arg("name"))
      .def("mac",
           [](npu_sim::Simulator& simulator, const std::string& pe_name,
              const py::object& activation, const py::object& weight) {
             return mac_result_to_dict(simulator.mac(
                 pe_name, parse_pe_input(activation), parse_pe_input(weight)));
           },
           py::arg("pe_name"), py::arg("activation"), py::arg("weight"))
      .def("run_systolic_array",
           [](npu_sim::Simulator& simulator, const std::string& array_name,
              const py::object& activations, const py::object& weights,
              bool trace_enabled) {
             return systolic_result_to_dict(simulator.run_systolic_array(
                 array_name, parse_pe_input_matrix(activations),
                 parse_pe_input_matrix(weights), trace_enabled));
           },
           py::arg("array_name"), py::arg("activations"), py::arg("weights"),
           py::arg("trace_enabled") = true)
      .def("run_systolic_array_stream",
           [](npu_sim::Simulator& simulator, const std::string& array_name,
              const py::object& activation_batches,
              const py::object& weight_batches, bool trace_enabled) {
             return systolic_result_to_dict(
                 simulator.run_systolic_array_stream(
                     array_name, parse_pe_input_matrix_batch(activation_batches),
                     parse_pe_input_matrix_batch(weight_batches),
                     trace_enabled));
           },
           py::arg("array_name"), py::arg("activation_batches"),
           py::arg("weight_batches"), py::arg("trace_enabled") = true)
      .def("run",
           [](npu_sim::Simulator& simulator) {
             return stats_to_dict(simulator.run());
           })
      .def("run_until",
           [](npu_sim::Simulator& simulator, npu_sim::Cycle max_cycle) {
             return stats_to_dict(simulator.run_until(max_cycle));
           },
           py::arg("max_cycle"))
      .def("stats",
           [](const npu_sim::Simulator& simulator) {
             return stats_to_dict(simulator.stats());
           });
}
