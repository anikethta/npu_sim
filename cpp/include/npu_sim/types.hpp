#pragma once

#include <cstdint>
#include <vector>

namespace npu_sim {

using Cycle = std::uint64_t;

enum class LogicValue {
  Zero,
  One,
  X,
  Z,
};

using LogicVector = std::vector<LogicValue>;

}  // namespace npu_sim
