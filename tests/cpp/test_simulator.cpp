#include "npu_sim/backend/systemc/systemc_backend.hpp"
#include "npu_sim/modules/processing_element.hpp"
#include "npu_sim/modules/sram_scratchpad.hpp"
#include "npu_sim/simulator.hpp"

#include <cmath>
#include <cstdlib>
#include <iostream>
#include <map>
#include <string>
#include <stdexcept>

namespace {

void expect(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::exit(1);
  }
}

template <typename Exception, typename Function>
void expect_throws(Function function, const std::string& message) {
  try {
    function();
  } catch (const Exception&) {
    return;
  } catch (...) {
    std::cerr << message << ": threw unexpected exception type\n";
    std::exit(1);
  }

  std::cerr << message << ": did not throw\n";
  std::exit(1);
}

npu_sim::SRAMScratchpadConfig small_sram_config(std::string name) {
  return npu_sim::SRAMScratchpadConfig{
      name,
      1,
      0,
      0,
      16,
      4,
      4,
      2,
      3,
  };
}

npu_sim::NumericFormatConfig signed_int_format(std::uint64_t bits) {
  return npu_sim::NumericFormatConfig{
      npu_sim::NumericFormatKind::SignedInt,
      bits,
      0,
      0,
  };
}

npu_sim::NumericFormatConfig unsigned_int_format(std::uint64_t bits) {
  return npu_sim::NumericFormatConfig{
      npu_sim::NumericFormatKind::UnsignedInt,
      bits,
      0,
      0,
  };
}

npu_sim::NumericFormatConfig float_format(std::uint64_t bits,
                                          std::uint64_t exponent_bits,
                                          std::uint64_t mantissa_bits) {
  return npu_sim::NumericFormatConfig{
      npu_sim::NumericFormatKind::Float,
      bits,
      exponent_bits,
      mantissa_bits,
  };
}

npu_sim::ProcessingElementConfig pe_config(
    std::string name, npu_sim::NumericFormatConfig activation_format,
    npu_sim::NumericFormatConfig weight_format,
    npu_sim::NumericFormatConfig accumulator_format, npu_sim::Cycle latency,
    npu_sim::PEDataflowMode dataflow_mode =
        npu_sim::PEDataflowMode::OutputStationary) {
  npu_sim::ProcessingElementConfig config;
  config.name = name;
  config.mac_latency_cycles = latency;
  config.dataflow_mode = dataflow_mode;
  config.activation.format = activation_format;
  config.weight.format = weight_format;
  config.accumulator = accumulator_format;
  return config;
}

npu_sim::SystolicArrayConfig systolic_array_config(
    std::string name,
    npu_sim::PEDataflowMode dataflow_mode =
        npu_sim::PEDataflowMode::OutputStationary) {
  npu_sim::SystolicArrayConfig config;
  config.name = name;
  config.size = 2;
  config.mac_latency_cycles = 1;
  config.dataflow_mode = dataflow_mode;
  config.activation.format = signed_int_format(3);
  config.weight.format = signed_int_format(3);
  config.accumulator = signed_int_format(32);
  return config;
}

npu_sim::SystolicArrayConfig requantized_systolic_array_config(
    std::string name) {
  npu_sim::SystolicArrayConfig config = systolic_array_config(name);
  config.requantization.enabled = true;
  config.requantization.target = signed_int_format(4);
  config.requantization.scale_multiplier.per_column = true;
  config.requantization.scale_multiplier.values = {3.0, 5.0};
  config.requantization.shift.per_column = true;
  config.requantization.shift.values = {1.0, 2.0};
  config.bias.enabled = true;
  config.bias.bias.per_column = true;
  config.bias.bias.values = {0.0, -1.0};
  return config;
}

npu_sim::SystolicArrayConfig after_vpu_requantized_systolic_array_config(
    std::string name) {
  npu_sim::SystolicArrayConfig config = requantized_systolic_array_config(name);
  config.output_vpu = "vpu0";
  config.requantization.placement = "after_vpu";
  return config;
}

npu_sim::SystolicArrayConfig inside_pe_bias_systolic_array_config(
    std::string name) {
  npu_sim::SystolicArrayConfig config = requantized_systolic_array_config(name);
  config.bias.placement = "inside_pe";
  return config;
}

npu_sim::SystolicArrayConfig inside_pe_bias_weight_stationary_config(
    std::string name) {
  npu_sim::SystolicArrayConfig config =
      systolic_array_config(name, npu_sim::PEDataflowMode::WeightStationary);
  config.bias.enabled = true;
  config.bias.placement = "inside_pe";
  config.bias.bias.per_column = true;
  config.bias.bias.values = {0.0, -1.0};
  return config;
}

npu_sim::VectorProcessingUnitConfig vpu_config(std::string name) {
  npu_sim::VectorProcessingUnitConfig config;
  config.name = name;
  config.enabled = true;
  config.lanes = 2;
  config.latency_cycles = 1;
  return config;
}

npu_sim::PEInputValue pe_input(std::int64_t value) {
  npu_sim::PEInputValue input;
  input.is_float = false;
  input.integer_value = value;
  input.floating_value = static_cast<double>(value);
  return input;
}

npu_sim::PEInputMatrix small_activation_matrix() {
  return {
      {pe_input(1), pe_input(2)},
      {pe_input(3), pe_input(1)},
  };
}

npu_sim::PEInputMatrix small_weight_matrix() {
  return {
      {pe_input(2), pe_input(1)},
      {pe_input(1), pe_input(-1)},
  };
}

npu_sim::Cycle current_cycle_from_systemc() {
  return static_cast<npu_sim::Cycle>(
      std::llround(sc_core::sc_time_stamp().to_seconds() / 1e-9));
}

void expect_logic(npu_sim::LogicValue actual, npu_sim::LogicValue expected,
                  const std::string& message) {
  expect(actual == expected, message);
}

void test_sram_scratchpad_reports_timing_and_capacity() {
  npu_sim::SRAMScratchpad scratchpad(
      "test_local_sram",
      npu_sim::SRAMScratchpadConfig{
          "local_sram",
          1,
          2,
          3,
          256,
          8,
          32,
          2,
          3,
      });

  expect(scratchpad.name() == "local_sram", "scratchpad name should match");
  expect(scratchpad.capacity_bits() == 8192,
         "scratchpad capacity should be word_size * num_words");
  expect(scratchpad.total_ports() == 6, "scratchpad should sum all ports");
  expect(scratchpad.read_latency_cycles() == 2,
         "scratchpad read latency should match config");
  expect(scratchpad.write_latency_cycles() == 3,
         "scratchpad write latency should match config");
}

void test_sram_rows_default_to_x(npu_sim::SRAMScratchpad& scratchpad) {
  const npu_sim::LogicVector row = scratchpad.read_word(0);

  expect(row.size() == 16, "read row should match configured word size");
  for (npu_sim::LogicValue bit : row) {
    expect_logic(bit, npu_sim::LogicValue::X,
                 "default SRAM bit should be X");
  }
  expect(scratchpad.read_count() == 1, "read count should increment");
}

void test_sram_write_then_read_returns_updated_bits(
    npu_sim::SRAMScratchpad& scratchpad) {
  const npu_sim::LogicVector data{
      npu_sim::LogicValue::One,
      npu_sim::LogicValue::Zero,
      npu_sim::LogicValue::One,
      npu_sim::LogicValue::One,
  };

  scratchpad.write_word(1, 4, data);
  const npu_sim::LogicVector row = scratchpad.read_word(1);

  expect_logic(row[0], npu_sim::LogicValue::X, "untouched bit should stay X");
  expect_logic(row[4], npu_sim::LogicValue::One, "written bit 0 should match");
  expect_logic(row[5], npu_sim::LogicValue::Zero, "written bit 1 should match");
  expect_logic(row[6], npu_sim::LogicValue::One, "written bit 2 should match");
  expect_logic(row[7], npu_sim::LogicValue::One, "written bit 3 should match");
  expect_logic(row[8], npu_sim::LogicValue::X,
               "bit after write range should stay X");
  expect(scratchpad.write_count() == 1, "write count should increment");
  expect(scratchpad.read_count() == 1, "read count should increment");
}

void test_sram_four_state_values_round_trip(npu_sim::SRAMScratchpad& scratchpad) {
  scratchpad.write_word(0, 0,
                        npu_sim::LogicVector{
                            npu_sim::LogicValue::Zero,
                            npu_sim::LogicValue::One,
                            npu_sim::LogicValue::X,
                            npu_sim::LogicValue::Z,
                        });

  const npu_sim::LogicVector row = scratchpad.read_word(0);

  expect_logic(row[0], npu_sim::LogicValue::Zero, "Zero should round-trip");
  expect_logic(row[1], npu_sim::LogicValue::One, "One should round-trip");
  expect_logic(row[2], npu_sim::LogicValue::X, "X should round-trip");
  expect_logic(row[3], npu_sim::LogicValue::Z, "Z should round-trip");
}

void test_sram_partial_write_preserves_existing_bits(
    npu_sim::SRAMScratchpad& scratchpad) {
  scratchpad.write_word(
      0, 0,
      npu_sim::LogicVector{
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
          npu_sim::LogicValue::One,
      });
  scratchpad.write_word(0, 4,
                        npu_sim::LogicVector{
                            npu_sim::LogicValue::Zero,
                            npu_sim::LogicValue::Zero,
                            npu_sim::LogicValue::Zero,
                            npu_sim::LogicValue::Zero,
                        });

  const npu_sim::LogicVector row = scratchpad.read_word(0);

  expect_logic(row[0], npu_sim::LogicValue::One,
               "first written range should remain set");
  expect_logic(row[3], npu_sim::LogicValue::One,
               "first written range should remain set");
  expect_logic(row[4], npu_sim::LogicValue::Zero,
               "partial write should clear selected bit");
  expect_logic(row[7], npu_sim::LogicValue::Zero,
               "partial write should clear selected bit");
  expect_logic(row[8], npu_sim::LogicValue::X,
               "untouched range should remain X");
}

void test_sram_rejects_invalid_transactions(npu_sim::SRAMScratchpad& scratchpad) {
  expect_throws<std::out_of_range>(
      [&scratchpad] { scratchpad.read_word(4); },
      "read past configured rows should fail");
  expect_throws<std::invalid_argument>(
      [&scratchpad] {
        scratchpad.write_word(
            0, 2,
            npu_sim::LogicVector{
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
            });
      },
      "unaligned write offset should fail");
  expect_throws<std::out_of_range>(
      [&scratchpad] {
        scratchpad.write_word(
            0, 12,
            npu_sim::LogicVector{
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
                npu_sim::LogicValue::One,
            });
      },
      "write past end of row should fail");
}

void test_sram_transactions_advance_systemc_time(
    npu_sim::SRAMScratchpad& scratchpad) {
  const npu_sim::Cycle start_cycle = current_cycle_from_systemc();

  scratchpad.write_word(0, 0,
                        npu_sim::LogicVector{
                            npu_sim::LogicValue::One,
                            npu_sim::LogicValue::Zero,
                            npu_sim::LogicValue::One,
                            npu_sim::LogicValue::Zero,
                        });
  const npu_sim::Cycle after_write_cycle = current_cycle_from_systemc();
  scratchpad.read_word(0);
  const npu_sim::Cycle after_read_cycle = current_cycle_from_systemc();

  expect(after_write_cycle == start_cycle + 3,
         "write should advance by configured write latency");
  expect(after_read_cycle == after_write_cycle + 2,
         "read should advance by configured read latency");
  expect(scratchpad.last_completion_cycle() == after_read_cycle,
         "last completion cycle should track last transaction");
}

void test_systemc_backend_advances_cycles() {
  npu_sim::SystemCBackendConfig config;
  config.core_count = 2;
  npu_sim::SystemCBackend backend(config);

  backend.run_until(5);

  expect(backend.current_cycle() == 5,
         "backend should advance to requested cycle");
  expect(backend.event_count() == 0, "backend has no internal events yet");
  expect(backend.component_count() == 2,
         "backend component count should track core count");
}

void test_pe_accepts_parameterized_integer_formats(
    npu_sim::ProcessingElement& int3_pe, npu_sim::ProcessingElement& int4_pe,
    npu_sim::ProcessingElement& int16_pe) {
  const npu_sim::PEValue int3_result = int3_pe.mac(
      npu_sim::PEValue::integer(signed_int_format(3), 3),
      npu_sim::PEValue::integer(signed_int_format(3), -2));
  expect(int3_result.integer_value() == -6, "INT3 MAC should accumulate");

  const npu_sim::PEValue int4_result = int4_pe.mac(
      npu_sim::PEValue::integer(signed_int_format(4), -4),
      npu_sim::PEValue::integer(signed_int_format(4), -2));
  expect(int4_result.integer_value() == 8, "INT4 MAC should accumulate");

  const npu_sim::PEValue int16_result = int16_pe.mac(
      npu_sim::PEValue::integer(signed_int_format(16), 200),
      npu_sim::PEValue::integer(signed_int_format(16), -3));
  expect(int16_result.integer_value() == -600, "INT16 MAC should accumulate");
}

void test_pe_unsigned_formats_and_range_validation(
    npu_sim::ProcessingElement& uint3_pe) {
  const npu_sim::PEValue result = uint3_pe.mac(
      npu_sim::PEValue::integer(unsigned_int_format(3), 7),
      npu_sim::PEValue::integer(unsigned_int_format(3), 2));
  expect(result.integer_value() == 14, "UINT3 MAC should accumulate");

  expect_throws<std::out_of_range>(
      [] { npu_sim::PEValue::integer(unsigned_int_format(3), 8); },
      "UINT3 should reject values above 7");
  expect_throws<std::out_of_range>(
      [] { npu_sim::PEValue::integer(signed_int_format(3), 4); },
      "INT3 should reject values above 3");
  expect_throws<std::out_of_range>(
      [] { npu_sim::PEValue::integer(signed_int_format(3), -5); },
      "INT3 should reject values below -4");
}

void test_pe_multiple_macs_accumulate(npu_sim::ProcessingElement& pe) {
  pe.mac(npu_sim::PEValue::integer(signed_int_format(8), 2),
         npu_sim::PEValue::integer(signed_int_format(8), 3));
  const npu_sim::PEValue result = pe.mac(
      npu_sim::PEValue::integer(signed_int_format(8), -1),
      npu_sim::PEValue::integer(signed_int_format(8), 4));

  expect(result.integer_value() == 2, "multiple MACs should accumulate");
  expect(pe.mac_count() == 2, "MAC count should track operations");
}

void test_pe_mac_advances_systemc_time(npu_sim::ProcessingElement& pe) {
  const npu_sim::Cycle start_cycle = current_cycle_from_systemc();

  pe.mac(npu_sim::PEValue::integer(signed_int_format(8), 2),
         npu_sim::PEValue::integer(signed_int_format(8), 3));
  const npu_sim::Cycle after_mac_cycle = current_cycle_from_systemc();

  expect(after_mac_cycle == start_cycle + 5,
         "PE MAC should advance by configured latency");
  expect(pe.last_completion_cycle() == after_mac_cycle,
         "PE last completion cycle should track MAC completion");
}

void test_pe_float_behavioral_mac(npu_sim::ProcessingElement& pe) {
  const npu_sim::NumericFormatConfig fp16 = float_format(16, 5, 10);
  const npu_sim::PEValue result =
      pe.mac(npu_sim::PEValue::floating(fp16, 1.5),
             npu_sim::PEValue::floating(fp16, 2.0));

  expect(result.is_float(), "floating MAC should produce floating accumulator");
  expect(std::abs(result.floating_value() - 3.0) < 1e-9,
         "floating MAC should use behavioral double value");
}

void test_pe_pass_through_returns_payload_unchanged(
    const npu_sim::ProcessingElement& pe) {
  const npu_sim::PEValue activation =
      npu_sim::PEValue::integer(signed_int_format(8), -7);
  const npu_sim::PEValue weight =
      npu_sim::PEValue::integer(signed_int_format(8), 5);
  const npu_sim::PEValue result =
      npu_sim::PEValue::integer(signed_int_format(16), -35);

  expect(pe.pass_activation(activation).integer_value() == -7,
         "activation pass-through should preserve value");
  expect(pe.pass_weight(weight).integer_value() == 5,
         "weight pass-through should preserve value");
  expect(pe.pass_result(result).integer_value() == -35,
         "result pass-through should preserve value");
}

void test_pe_accepts_dataflow_modes(
    const npu_sim::ProcessingElement& output_stationary_pe,
    const npu_sim::ProcessingElement& weight_stationary_pe) {
  expect(output_stationary_pe.dataflow_mode() ==
             npu_sim::PEDataflowMode::OutputStationary,
         "PE should accept output-stationary mode");
  expect(weight_stationary_pe.dataflow_mode() ==
             npu_sim::PEDataflowMode::WeightStationary,
         "PE should accept weight-stationary mode");
}

void test_simulator_component_count_includes_scratchpads(
    const npu_sim::Simulator& simulator) {
  expect(simulator.stats().component_count == 6,
         "component count should include cores and scratchpads");
}

void test_systolic_array_output_stationary_trace(
    npu_sim::SystolicArray& array) {
  const npu_sim::Cycle start_cycle = current_cycle_from_systemc();
  const npu_sim::SystolicArrayResult result = array.run_matrix_multiply(
      small_activation_matrix(), small_weight_matrix(), true);

  expect(result.outputs[0][0].integer_value == 4,
         "systolic output C00 should match matmul");
  expect(result.outputs[0][1].integer_value == -1,
         "systolic output C01 should match matmul");
  expect(result.outputs[1][0].integer_value == 7,
         "systolic output C10 should match matmul");
  expect(result.outputs[1][1].integer_value == 2,
         "systolic output C11 should match matmul");
  expect(result.operation_count == 8,
         "2x2 systolic matmul should perform size^3 MACs");
  expect(result.current_cycle == start_cycle + 5,
         "output-stationary systolic run should advance to final emit cycle");

  std::uint64_t activation_inject_counts[3] = {0, 0, 0};
  bool saw_first_commit = false;
  bool saw_fifo_push = false;
  bool saw_deasserted_dequeue_request = false;
  bool saw_fifo_dequeue = false;
  std::map<npu_sim::Cycle, bool> fifo_push_by_write_cycle;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "inject_activation") {
      const npu_sim::Cycle relative_cycle = event.cycle - start_cycle;
      if (relative_cycle < 3) {
        ++activation_inject_counts[relative_cycle];
      }
    }
    if (event.event == "mac_commit" && event.cycle == start_cycle + 1) {
      saw_first_commit = true;
    }
    if (event.event == "spu_output_fifo_push") {
      saw_fifo_push = true;
      fifo_push_by_write_cycle[event.cycle + 1] = true;
      expect(event.lanes.size() == 2,
             "size-2 array should still bank raw SPU FIFO entries");
    }
    if (event.event == "spu_output_fifo_dequeue_request" &&
        !event.dequeue_asserted) {
      saw_deasserted_dequeue_request = true;
      expect(fifo_push_by_write_cycle[event.cycle],
             "SPU FIFO dequeue request should occur when the write is visible");
    }
    if (event.event == "spu_output_fifo_dequeue") {
      saw_fifo_dequeue = true;
    }
  }

  expect(activation_inject_counts[0] == 1,
         "first cycle should inject one activation");
  expect(activation_inject_counts[1] == 2,
         "second cycle should inject two activations");
  expect(activation_inject_counts[2] == 1,
         "third cycle should inject one activation");
  expect(saw_first_commit, "1-cycle MAC should commit one cycle after start");
  expect(saw_fifo_push, "raw SPU outputs should enqueue into the SPU FIFO");
  expect(saw_deasserted_dequeue_request,
         "array without an output VPU should deassert FIFO dequeue");
  expect(!saw_fifo_dequeue,
         "array without an output VPU should leave FIFO entries queued");
}

void test_systolic_array_weight_stationary_trace(
    npu_sim::SystolicArray& array) {
  const npu_sim::Cycle start_cycle = current_cycle_from_systemc();
  const npu_sim::SystolicArrayResult result = array.run_matrix_multiply(
      small_activation_matrix(), small_weight_matrix(), true);

  expect(result.outputs[0][0].integer_value == 4,
         "weight-stationary output C00 should match matmul");
  expect(result.outputs[1][1].integer_value == 2,
         "weight-stationary output C11 should match matmul");
  expect(result.current_cycle > start_cycle + 4,
         "weight-stationary run should include load timing");

  bool saw_weight_load = false;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "weight_load") {
      saw_weight_load = true;
    }
  }
  expect(saw_weight_load, "weight-stationary trace should include loads");
}

void test_systolic_array_weight_stationary_reuses_resident_weights(
    npu_sim::SystolicArray& array) {
  const npu_sim::PEInputMatrixBatch activation_batches{
      small_activation_matrix(),
      {
          {pe_input(2), pe_input(0)},
          {pe_input(1), pe_input(-1)},
      },
  };
  const npu_sim::PEInputMatrixBatch weight_batches{small_weight_matrix()};
  const npu_sim::SystolicArrayResult result =
      array.run_matrix_multiply_stream(activation_batches, weight_batches, true);

  expect(result.output_batches[0][0][0].integer_value == 4,
         "first activation batch should use initially loaded weights");
  expect(result.output_batches[1][0][0].integer_value == 4,
         "second activation batch should reuse resident C00 weight path");
  expect(result.output_batches[1][0][1].integer_value == 2,
         "second activation batch should reuse resident C01 weight path");

  bool saw_reuse = false;
  bool saw_batch_one_weight_fifo = false;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "stationary_weight_reuse" && event.batch == 1) {
      saw_reuse = true;
    }
    if ((event.event == "weight_fifo_push" ||
         event.event == "weight_fifo_pop") &&
        event.batch == 1) {
      saw_batch_one_weight_fifo = true;
    }
  }

  expect(saw_reuse,
         "weight-stationary stream should trace resident weight reuse");
  expect(!saw_batch_one_weight_fifo,
         "resident weight reuse should not consume another weight FIFO batch");
}

void test_systolic_array_requantizes_outputs(
    npu_sim::SystolicArray& array) {
  const npu_sim::SystolicArrayResult result = array.run_matrix_multiply(
      small_activation_matrix(), small_weight_matrix(), true);

  expect(result.outputs[0][0].integer_value == 6,
         "requantized C00 should include multiply, rounding, and shift");
  expect(result.outputs[0][1].integer_value == -2,
         "requantized C01 should use per-column bias and arithmetic shift");
  expect(result.outputs[1][0].integer_value == 7,
         "requantized C10 should saturate to signed INT4 max");
  expect(result.outputs[1][1].integer_value == 1,
         "requantized C11 should use per-column registers");
  expect(result.outputs[0][0].format.bits == 4,
         "requantized outputs should use target format");

  bool saw_requant_shift = false;
  bool saw_requant_saturate = false;
  bool saw_bias_add = false;
  bool saw_requantized_output_emit = false;
  npu_sim::Cycle pre_requant_cycle = 0;
  npu_sim::Cycle output_emit_cycle = 0;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "pre_requant_output" && event.row == 1 &&
        event.col == 0 && event.value == 7.0) {
      pre_requant_cycle = event.cycle;
    }
    if (event.event == "requant_shift" && event.col == 1 && event.shift == 2) {
      saw_requant_shift = true;
    }
    if (event.event == "bias_add" && event.col == 1 && event.bias == -1.0 &&
        event.bias_format_bits == 16) {
      saw_bias_add = true;
    }
    if (event.event == "requant_saturate" && event.row == 1 &&
        event.col == 0 && event.value == 7.0) {
      saw_requant_saturate = true;
    }
    if (event.event == "output_emit" && event.row == 1 && event.col == 0 &&
        event.value == 7.0) {
      saw_requantized_output_emit = true;
      output_emit_cycle = event.cycle;
    }
  }
  expect(pre_requant_cycle != 0,
         "PE should emit pre-quantized accumulator value before requantization");
  expect(output_emit_cycle == pre_requant_cycle + 6,
         "requantized output should emerge after the implicit FIFO write plus five pipeline cycles");
  expect(saw_bias_add, "bias adder trace should include high-precision bias");
  expect(saw_requant_shift, "requantization trace should include shift metadata");
  expect(saw_requant_saturate, "requantization trace should include saturation");
  expect(saw_requantized_output_emit,
         "output_emit should carry final requantized value");
}

void test_systolic_array_can_requantize_after_vpu(
    npu_sim::SystolicArray& array, npu_sim::VectorProcessingUnit& vpu) {
  array.attach_output_vpu(&vpu);
  const npu_sim::SystolicArrayResult result = array.run_matrix_multiply(
      small_activation_matrix(), small_weight_matrix(), true);

  expect(result.outputs[0][0].integer_value == 6,
         "post-VPU requantized C00 should match pass-through VPU output");
  expect(result.outputs[1][0].integer_value == 7,
         "post-VPU requantized C10 should still saturate");

  bool saw_spu_fifo = false;
  bool saw_vpu = false;
  bool saw_after_vpu_requant = false;
  bool saw_pre_requant = false;
  bool saw_invalid_fifo_lane = false;
  bool saw_fifo_dequeue_request = false;
  bool saw_fifo_dequeue = false;
  bool saw_fifo_empty = false;
  std::map<npu_sim::Cycle, std::uint64_t> fifo_pushes_by_cycle;
  std::map<npu_sim::Cycle, bool> fifo_push_by_write_cycle;
  std::map<npu_sim::Cycle, bool> dequeue_request_by_cycle;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "spu_output_fifo_push") {
      saw_spu_fifo = true;
      ++fifo_pushes_by_cycle[event.cycle];
      fifo_push_by_write_cycle[event.cycle + 1] = true;
      expect(event.lanes.size() == 2,
             "size-2 array should emit two-lane SPU FIFO entries");
      for (const npu_sim::SPUOutputFIFOLane& lane : event.lanes) {
        if (!lane.valid && lane.value == 0.0) {
          saw_invalid_fifo_lane = true;
        }
      }
    }
    if (event.event == "spu_output_fifo_dequeue_request") {
      saw_fifo_dequeue_request = true;
      expect(event.dequeue_asserted,
             "dequeue request should assert the SPU FIFO dequeue signal");
      expect(fifo_push_by_write_cycle[event.cycle],
             "SPU FIFO dequeue request should wait one cycle after push");
      dequeue_request_by_cycle[event.cycle] = true;
    }
    if (event.event == "spu_output_fifo_dequeue") {
      saw_fifo_dequeue = true;
      expect(event.dequeue_asserted,
             "dequeue event should carry asserted dequeue signal");
      expect(dequeue_request_by_cycle[event.cycle],
             "SPU FIFO dequeue should follow a dequeue request");
    }
    if (event.event == "spu_output_fifo_empty") {
      saw_fifo_empty = true;
    }
    if (event.event == "vpu_pass_through") {
      saw_vpu = true;
    }
    if (event.event == "requant_shift" && event.placement == "after_vpu") {
      saw_after_vpu_requant = true;
    }
    if (event.event == "pre_requant_output") {
      saw_pre_requant = true;
    }
  }

  expect(saw_spu_fifo, "SPU output should pass through the output FIFO");
  for (const auto& [cycle, count] : fifo_pushes_by_cycle) {
    expect(count <= 1, "SPU FIFO should enqueue at most one entry per cycle");
  }
  expect(saw_invalid_fifo_lane,
         "SPU FIFO should include zero/void lanes for partial entries");
  expect(saw_fifo_dequeue_request,
         "SPU FIFO should trace explicit dequeue requests");
  expect(saw_fifo_dequeue, "SPU FIFO should trace explicit dequeues");
  expect(!saw_fifo_empty,
         "SPU FIFO should not dequeue when no entry is available");
  expect(saw_vpu, "post-SPU output should pass through the VPU");
  expect(saw_after_vpu_requant,
         "requantization should be traceable after the VPU");
  expect(!saw_pre_requant,
         "systolic trace should not mark pre-requant output before VPU");
}

void test_systolic_array_can_seed_bias_inside_pe(
    npu_sim::SystolicArray& array) {
  const npu_sim::SystolicArrayResult result = array.run_matrix_multiply(
      small_activation_matrix(), small_weight_matrix(), true);

  expect(result.outputs[0][0].integer_value == 6,
         "inside-PE biased C00 should match post-array bias/requantization");
  expect(result.outputs[0][1].integer_value == -2,
         "inside-PE biased C01 should include per-column bias");
  expect(result.outputs[1][0].integer_value == 7,
         "inside-PE biased C10 should still saturate");
  expect(result.outputs[1][1].integer_value == 1,
         "inside-PE biased C11 should match post-array pipeline semantics");

  bool saw_bias_fifo_push = false;
  bool saw_bias_fifo_pop = false;
  bool saw_bias_register_load = false;
  bool saw_bias_accumulator_seed = false;
  bool saw_bias_add = false;
  bool saw_pre_requant_biased_value = false;
  npu_sim::Cycle base_cycle = 0;
  npu_sim::Cycle col0_push_cycle = 0;
  npu_sim::Cycle col1_push_cycle = 0;
  npu_sim::Cycle pe11_seed_cycle = 0;
  std::uint64_t bias_fifo_push_count = 0;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "mac_start" && event.row == 0 && event.col == 0 &&
        event.k == 0) {
      base_cycle = event.cycle;
    }
    if (event.event == "bias_fifo_push" && event.placement == "inside_pe") {
      saw_bias_fifo_push = true;
      ++bias_fifo_push_count;
      if (event.row == -1 && event.col == 0) {
        col0_push_cycle = event.cycle;
      }
      if (event.row == -1 && event.col == 1) {
        col1_push_cycle = event.cycle;
      }
    }
    if (event.event == "bias_fifo_pop" && event.placement == "inside_pe") {
      saw_bias_fifo_pop = true;
    }
    if (event.event == "bias_register_load" &&
        event.placement == "inside_pe") {
      saw_bias_register_load = true;
    }
    if (event.event == "bias_accumulator_seed" &&
        event.placement == "inside_pe" && event.col == 1 &&
        event.bias == -1.0 && event.bias_format_bits == 16) {
      saw_bias_accumulator_seed = true;
      if (event.row == 1) {
        pe11_seed_cycle = event.cycle;
      }
    }
    if (event.event == "bias_add") {
      saw_bias_add = true;
    }
    if (event.event == "pre_requant_output" && event.row == 1 &&
        event.col == 1 && event.value == 1.0) {
      saw_pre_requant_biased_value = true;
    }
  }

  expect(saw_bias_fifo_push,
         "inside-PE bias should trace bias FIFO ingress");
  expect(saw_bias_fifo_pop,
         "inside-PE bias should trace bias FIFO consumption");
  expect(bias_fifo_push_count == 2,
         "inside-PE bias should push one top-edge bias per column");
  expect(col0_push_cycle == base_cycle,
         "column 0 bias should enter from the top at the base cycle");
  expect(col1_push_cycle == base_cycle + 1,
         "column 1 bias should enter from the top one cycle later");
  expect(pe11_seed_cycle == base_cycle + 2,
         "PE(1,1) should seed when the streamed bias reaches it");
  expect(saw_bias_register_load,
         "inside-PE bias should trace PE-local register load");
  expect(saw_bias_accumulator_seed,
         "inside-PE bias should trace accumulator seeding metadata");
  expect(!saw_bias_add,
         "inside-PE bias should not use the post-array bias adder");
  expect(saw_pre_requant_biased_value,
         "requantization should see already-biased PE accumulator values");
}

void test_weight_stationary_array_can_seed_bias_inside_pe(
    npu_sim::SystolicArray& array) {
  const npu_sim::PEInputMatrixBatch activation_batches{
      small_activation_matrix(),
      {
          {pe_input(2), pe_input(0)},
          {pe_input(1), pe_input(-1)},
      },
  };
  const npu_sim::PEInputMatrixBatch weight_batches{small_weight_matrix()};
  const npu_sim::SystolicArrayResult result =
      array.run_matrix_multiply_stream(activation_batches, weight_batches, true);

  expect(result.output_batches[0][0][0].integer_value == 4,
         "inside-PE WS bias should preserve C00");
  expect(result.output_batches[0][0][1].integer_value == -2,
         "inside-PE WS bias should adjust C01");
  expect(result.output_batches[1][0][0].integer_value == 4,
         "inside-PE WS bias should seed reused-weight batch C00");
  expect(result.output_batches[1][0][1].integer_value == 1,
         "inside-PE WS bias should seed reused-weight batch C01");

  bool saw_reuse = false;
  bool saw_batch_one_bias_seed = false;
  bool saw_bias_add = false;
  npu_sim::Cycle batch_one_base_cycle = 0;
  npu_sim::Cycle batch_one_col1_push_cycle = 0;
  npu_sim::Cycle batch_one_pe11_seed_cycle = 0;
  for (const npu_sim::SystolicTraceEvent& event : result.trace) {
    if (event.event == "stationary_weight_reuse" && event.batch == 1) {
      saw_reuse = true;
    }
    if (event.event == "mac_start" && event.batch == 1 && event.row == 0 &&
        event.col == 0 && event.k == 0) {
      batch_one_base_cycle = event.cycle;
    }
    if (event.event == "bias_fifo_push" && event.batch == 1 &&
        event.row == -1 && event.col == 1) {
      batch_one_col1_push_cycle = event.cycle;
    }
    if (event.event == "bias_accumulator_seed" && event.batch == 1 &&
        event.placement == "inside_pe") {
      saw_batch_one_bias_seed = true;
      if (event.row == 1 && event.col == 1) {
        batch_one_pe11_seed_cycle = event.cycle;
      }
    }
    if (event.event == "bias_add") {
      saw_bias_add = true;
    }
  }

  expect(saw_reuse, "weight-stationary test should reuse resident weights");
  expect(saw_batch_one_bias_seed,
         "inside-PE bias should seed each activation batch");
  expect(batch_one_col1_push_cycle == batch_one_base_cycle + 1,
         "weight-stationary bias should stream one column per cycle");
  expect(batch_one_pe11_seed_cycle == batch_one_base_cycle + 2,
         "weight-stationary bias should propagate down from the top");
  expect(!saw_bias_add,
         "inside-PE weight-stationary bias should skip post-array bias add");
}

}  // namespace

int main() {
  npu_sim::SRAMScratchpad zero_sram("test_zero_sram",
                                    small_sram_config("zero_sram"));
  npu_sim::SRAMScratchpad rw_sram("test_rw_sram",
                                  small_sram_config("rw_sram"));
  npu_sim::SRAMScratchpad partial_sram("test_partial_sram",
                                       small_sram_config("partial_sram"));
  npu_sim::SRAMScratchpad four_state_sram(
      "test_four_state_sram", small_sram_config("four_state_sram"));
  npu_sim::SRAMScratchpad invalid_sram("test_invalid_sram",
                                       small_sram_config("invalid_sram"));
  npu_sim::SRAMScratchpad timing_sram("test_timing_sram",
                                      small_sram_config("timing_sram"));
  npu_sim::ProcessingElement int3_pe(
      "test_int3_pe",
      pe_config("int3_pe", signed_int_format(3), signed_int_format(3),
                signed_int_format(16), 1));
  npu_sim::ProcessingElement int4_pe(
      "test_int4_pe",
      pe_config("int4_pe", signed_int_format(4), signed_int_format(4),
                signed_int_format(16), 1));
  npu_sim::ProcessingElement int16_pe(
      "test_int16_pe",
      pe_config("int16_pe", signed_int_format(16), signed_int_format(16),
                signed_int_format(32), 1));
  npu_sim::ProcessingElement uint3_pe(
      "test_uint3_pe",
      pe_config("uint3_pe", unsigned_int_format(3), unsigned_int_format(3),
                signed_int_format(16), 1));
  npu_sim::ProcessingElement accumulate_pe(
      "test_accumulate_pe",
      pe_config("accumulate_pe", signed_int_format(8), signed_int_format(8),
                signed_int_format(32), 1));
  npu_sim::ProcessingElement timing_pe(
      "test_timing_pe",
      pe_config("timing_pe", signed_int_format(8), signed_int_format(8),
                signed_int_format(32), 5));
  npu_sim::ProcessingElement fp_pe(
      "test_fp_pe",
      pe_config("fp_pe", float_format(16, 5, 10), float_format(16, 5, 10),
                float_format(32, 8, 23), 1));
  npu_sim::ProcessingElement pass_pe(
      "test_pass_pe",
      pe_config("pass_pe", signed_int_format(8), signed_int_format(8),
                signed_int_format(16), 1));
  npu_sim::ProcessingElement output_stationary_pe(
      "test_output_stationary_pe",
      pe_config("output_stationary_pe", signed_int_format(8),
                signed_int_format(8), signed_int_format(32), 1,
                npu_sim::PEDataflowMode::OutputStationary));
  npu_sim::ProcessingElement weight_stationary_pe(
      "test_weight_stationary_pe",
      pe_config("weight_stationary_pe", signed_int_format(8),
                signed_int_format(8), signed_int_format(32), 1,
                npu_sim::PEDataflowMode::WeightStationary));
  npu_sim::SystolicArray output_stationary_array(
      "test_output_stationary_array",
      systolic_array_config("output_stationary_array"));
  npu_sim::SystolicArray weight_stationary_array(
      "test_weight_stationary_array",
      systolic_array_config("weight_stationary_array",
                            npu_sim::PEDataflowMode::WeightStationary));
  npu_sim::SystolicArray requantized_array(
      "test_requantized_array",
      requantized_systolic_array_config("requantized_array"));
  npu_sim::SystolicArray after_vpu_requantized_array(
      "test_after_vpu_requantized_array",
      after_vpu_requantized_systolic_array_config(
          "after_vpu_requantized_array"));
  npu_sim::SystolicArray inside_pe_bias_array(
      "test_inside_pe_bias_array",
      inside_pe_bias_systolic_array_config("inside_pe_bias_array"));
  npu_sim::SystolicArray inside_pe_bias_ws_array(
      "test_inside_pe_bias_ws_array",
      inside_pe_bias_weight_stationary_config("inside_pe_bias_ws_array"));
  npu_sim::VectorProcessingUnit vpu("test_vpu", vpu_config("vpu0"));
  npu_sim::SimulatorConfig simulator_config;
  simulator_config.core_count = 4;
  simulator_config.scratchpads.push_back(
      npu_sim::SRAMScratchpadConfig{"local_sram"});
  simulator_config.scratchpads.push_back(
      npu_sim::SRAMScratchpadConfig{"global_sram"});
  npu_sim::Simulator simulator(simulator_config);

  test_sram_scratchpad_reports_timing_and_capacity();
  test_simulator_component_count_includes_scratchpads(simulator);
  test_sram_rows_default_to_x(zero_sram);
  test_sram_write_then_read_returns_updated_bits(rw_sram);
  test_sram_four_state_values_round_trip(four_state_sram);
  test_sram_partial_write_preserves_existing_bits(partial_sram);
  test_sram_rejects_invalid_transactions(invalid_sram);
  test_sram_transactions_advance_systemc_time(timing_sram);
  test_pe_accepts_parameterized_integer_formats(int3_pe, int4_pe, int16_pe);
  test_pe_unsigned_formats_and_range_validation(uint3_pe);
  test_pe_multiple_macs_accumulate(accumulate_pe);
  test_pe_mac_advances_systemc_time(timing_pe);
  test_pe_float_behavioral_mac(fp_pe);
  test_pe_pass_through_returns_payload_unchanged(pass_pe);
  test_pe_accepts_dataflow_modes(output_stationary_pe, weight_stationary_pe);
  test_systolic_array_output_stationary_trace(output_stationary_array);
  test_systolic_array_weight_stationary_trace(weight_stationary_array);
  test_systolic_array_weight_stationary_reuses_resident_weights(
      weight_stationary_array);
  test_systolic_array_requantizes_outputs(requantized_array);
  test_systolic_array_can_requantize_after_vpu(after_vpu_requantized_array,
                                               vpu);
  test_systolic_array_can_seed_bias_inside_pe(inside_pe_bias_array);
  test_weight_stationary_array_can_seed_bias_inside_pe(
      inside_pe_bias_ws_array);
  test_systemc_backend_advances_cycles();
  return 0;
}
