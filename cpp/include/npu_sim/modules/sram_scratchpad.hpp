#pragma once

#include <cstdint>
#include <string>

#include <systemc>

#include "npu_sim/types.hpp"

namespace npu_sim {

struct SRAMScratchpadConfig {
  std::string name;
  std::uint64_t num_rw_ports{1};
  std::uint64_t num_r_ports{0};
  std::uint64_t num_w_ports{0};
  std::uint64_t word_size{256};
  std::uint64_t write_size{8};
  std::uint64_t num_words{32};
  Cycle read_latency_cycles{1};
  Cycle write_latency_cycles{1};
};

class SRAMScratchpad : public sc_core::sc_module {
 public:
  SRAMScratchpad(sc_core::sc_module_name module_name,
                 SRAMScratchpadConfig config);

  const std::string& name() const;
  std::uint64_t capacity_bits() const;
  std::uint64_t total_ports() const;
  std::uint64_t num_rw_ports() const;
  std::uint64_t num_r_ports() const;
  std::uint64_t num_w_ports() const;
  std::uint64_t word_size() const;
  std::uint64_t write_size() const;
  std::uint64_t num_words() const;
  Cycle read_latency_cycles() const;
  Cycle write_latency_cycles() const;
  std::uint64_t read_count() const;
  std::uint64_t write_count() const;
  Cycle last_completion_cycle() const;

  LogicVector read_word(std::uint64_t row_index);
  void write_word(std::uint64_t row_index, std::uint64_t write_offset_bits,
                  const LogicVector& data_bits);

 private:
  void advance_cycles(Cycle cycles);
  void validate_row_index(std::uint64_t row_index) const;
  void validate_write(std::uint64_t row_index,
                      std::uint64_t write_offset_bits,
                      const LogicVector& data_bits) const;

  SRAMScratchpadConfig config_;
  std::vector<LogicVector> storage_;
  std::uint64_t read_count_{0};
  std::uint64_t write_count_{0};
  Cycle last_completion_cycle_{0};
};

}  // namespace npu_sim
