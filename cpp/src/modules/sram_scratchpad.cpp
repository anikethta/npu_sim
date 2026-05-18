#include "npu_sim/modules/sram_scratchpad.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace npu_sim {

namespace {

void validate_config(const SRAMScratchpadConfig& config) {
  if (config.name.empty()) {
    throw std::invalid_argument("scratchpad name must be non-empty");
  }
  if (config.num_rw_ports + config.num_r_ports + config.num_w_ports == 0) {
    throw std::invalid_argument("scratchpad must have at least one port");
  }
  if (config.word_size == 0) {
    throw std::invalid_argument("scratchpad word_size must be positive");
  }
  if (config.write_size == 0) {
    throw std::invalid_argument("scratchpad write_size must be positive");
  }
  if (config.num_words == 0) {
    throw std::invalid_argument("scratchpad num_words must be positive");
  }
  if (config.write_size > config.word_size) {
    throw std::invalid_argument("scratchpad write_size must be <= word_size");
  }
  if (config.word_size % config.write_size != 0) {
    throw std::invalid_argument(
        "scratchpad word_size must be divisible by write_size");
  }
}

}  // namespace

SRAMScratchpad::SRAMScratchpad(sc_core::sc_module_name module_name,
                               SRAMScratchpadConfig config)
    : sc_core::sc_module(module_name), config_(std::move(config)) {
  validate_config(config_);
  storage_.resize(config_.num_words,
                  LogicVector(config_.word_size, LogicValue::X));
}

const std::string& SRAMScratchpad::name() const { return config_.name; }

std::uint64_t SRAMScratchpad::capacity_bits() const {
  return config_.word_size * config_.num_words;
}

std::uint64_t SRAMScratchpad::total_ports() const {
  return config_.num_rw_ports + config_.num_r_ports + config_.num_w_ports;
}

std::uint64_t SRAMScratchpad::num_rw_ports() const {
  return config_.num_rw_ports;
}

std::uint64_t SRAMScratchpad::num_r_ports() const {
  return config_.num_r_ports;
}

std::uint64_t SRAMScratchpad::num_w_ports() const {
  return config_.num_w_ports;
}

std::uint64_t SRAMScratchpad::word_size() const { return config_.word_size; }

std::uint64_t SRAMScratchpad::write_size() const { return config_.write_size; }

std::uint64_t SRAMScratchpad::num_words() const { return config_.num_words; }

Cycle SRAMScratchpad::read_latency_cycles() const {
  return config_.read_latency_cycles;
}

Cycle SRAMScratchpad::write_latency_cycles() const {
  return config_.write_latency_cycles;
}

std::uint64_t SRAMScratchpad::read_count() const { return read_count_; }

std::uint64_t SRAMScratchpad::write_count() const { return write_count_; }

Cycle SRAMScratchpad::last_completion_cycle() const {
  return last_completion_cycle_;
}

LogicVector SRAMScratchpad::read_word(std::uint64_t row_index) {
  validate_row_index(row_index);
  advance_cycles(config_.read_latency_cycles);
  ++read_count_;
  return storage_[row_index];
}

void SRAMScratchpad::write_word(std::uint64_t row_index,
                                std::uint64_t write_offset_bits,
                                const LogicVector& data_bits) {
  validate_write(row_index, write_offset_bits, data_bits);
  advance_cycles(config_.write_latency_cycles);
  std::copy(data_bits.begin(), data_bits.end(),
            storage_[row_index].begin() + write_offset_bits);
  ++write_count_;
}

void SRAMScratchpad::advance_cycles(Cycle cycles) {
  if (cycles > 0) {
    sc_core::sc_start(sc_core::sc_time(static_cast<double>(cycles),
                                       sc_core::SC_NS));
  }
  last_completion_cycle_ =
      static_cast<Cycle>(
          std::llround(sc_core::sc_time_stamp().to_seconds() / 1e-9));
}

void SRAMScratchpad::validate_row_index(std::uint64_t row_index) const {
  if (row_index >= config_.num_words) {
    throw std::out_of_range("scratchpad row index is out of range");
  }
}

void SRAMScratchpad::validate_write(std::uint64_t row_index,
                                    std::uint64_t write_offset_bits,
                                    const LogicVector& data_bits) const {
  validate_row_index(row_index);
  if (data_bits.empty()) {
    throw std::invalid_argument("scratchpad write data must be non-empty");
  }
  if (write_offset_bits % config_.write_size != 0) {
    throw std::invalid_argument(
        "scratchpad write offset must be aligned to write_size");
  }
  if (data_bits.size() % config_.write_size != 0) {
    throw std::invalid_argument(
        "scratchpad write data size must be a multiple of write_size");
  }
  if (write_offset_bits + data_bits.size() > config_.word_size) {
    throw std::out_of_range("scratchpad write extends past end of row");
  }
}

}  // namespace npu_sim
