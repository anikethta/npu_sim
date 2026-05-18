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
      const py::dict scratchpad = item.cast<py::dict>();
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
      native_config.scratchpads.push_back(std::move(scratchpad_config));
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
  return out;
}

py::dict systolic_result_to_dict(const npu_sim::SystolicArrayResult& result) {
  py::dict out;
  py::list outputs;
  for (const auto& row : result.outputs) {
    py::list output_row;
    for (const npu_sim::PEMacResult& value : row) {
      output_row.append(mac_result_to_dict(value)["value"]);
    }
    outputs.append(std::move(output_row));
  }
  py::list trace;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    trace.append(trace_event_to_dict(event));
  }
  out["outputs"] = std::move(outputs);
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
