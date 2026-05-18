# NPU Sim

A modular NPU simulation platform with an integrated Python frontend.

## Current Scaffold

- Python package: `npu_sim`
- Native backend: C++17, CMake, pybind11, SystemC
- Simulation model: SystemC-backed cycle-accurate simulation
- Public API shape: typed config objects plus an `NPU(config)` facade
- Native logic model: four-state `0`, `1`, `X`, `Z`
- Implemented SystemC modules:
  - SRAM scratchpad with bit-level storage, configurable ports/shape/latency,
    four-state startup `X`, reads, writes, and timing counters
  - Processing element with configurable MAC latency, parameterized numeric
    formats, output-stationary/weight-stationary config, local accumulator, and
    C++-only behavioral MAC tests
- C++ layout:
  - `cpp/include/npu_sim/simulator.hpp`: Python-facing simulator facade
  - `cpp/include/npu_sim/backend/systemc/`: SystemC runtime/backend internals
  - `cpp/include/npu_sim/modules/`: reusable SystemC hardware modules

## Development Setup

Install SystemC first and make its CMake package discoverable with
`CMAKE_PREFIX_PATH` if it ships CMake package files, or `PKG_CONFIG_PATH` if it
ships pkg-config metadata, as Homebrew does.

On this macOS environment, Homebrew installs SystemC under `/usr/local` as an
`x86_64` library, so use the `x86_64` CMake architecture flag shown below.

```bash
python3 -m venv .venv
export SYSTEMC_PREFIX="$(brew --prefix systemc)"
export PKG_CONFIG_PATH="$SYSTEMC_PREFIX/lib/pkgconfig"

CMAKE_ARGS="-DCMAKE_OSX_ARCHITECTURES=x86_64 -DPython3_EXECUTABLE=$(pwd)/.venv/bin/python" \
  .venv/bin/python -m pip install -e ".[dev]"

.venv/bin/python -m pytest
```

To run the C++ smoke tests through CMake directly:

```bash
export SYSTEMC_PREFIX="$(brew --prefix systemc)"
export PKG_CONFIG_PATH="$SYSTEMC_PREFIX/lib/pkgconfig"

cmake -S . -B build \
  -DCMAKE_OSX_ARCHITECTURES=x86_64 \
  -DPython3_EXECUTABLE="$(pwd)/.venv/bin/python" \
  -DCMAKE_PREFIX_PATH="$SYSTEMC_PREFIX" \
  -Dpybind11_DIR=$(.venv/bin/python -m pybind11 --cmakedir)
cmake --build build
ctest --test-dir build --output-on-failure
```

## Modeling Notes

- SystemC modules must be constructed before the first `sc_start`.
- `cpp/src/systemc_main.cpp` provides the embedded `sc_main` symbol needed when
  SystemC is loaded through the Python extension.
- SRAM rows initialize to `X`. Reads before writes should observe `X`, not `0`.
- Use `LogicValue` / `LogicVector` for native hardware state that can be
  unknown or high impedance.
- PE operand formats are parameterized:
  - `INT3`: signed integer, `bits=3`
  - `INT4`: signed integer, `bits=4`
  - `UINT8`: unsigned integer, `bits=8`
  - `FP8`: float, `bits=8`, with explicit exponent/mantissa metadata
  - `FP16`: float, `bits=16`, with explicit exponent/mantissa metadata
- PE floating-point behavior is currently host-`double` behavioral simulation,
  not bit-accurate FP8/FP16 encoding.
- Public Python config can instantiate SRAMs and PEs; PE MAC execution is still
  C++-only while the workload API is being designed.

## Minimal Python Example

```python
from npu_sim import (
    CoreConfig,
    NPU,
    NPUConfig,
    NumericFormatConfig,
    NumericFormatKind,
    PEDataflowMode,
    PEOperandConfig,
    ProcessingElementConfig,
    SRAMScratchpadConfig,
)

int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
acc = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=32)

config = NPUConfig(
    cores=CoreConfig(count=4),
    scratchpads=[
        SRAMScratchpadConfig(
            name="local_sram",
            num_rw_ports=1,
            num_r_ports=0,
            num_w_ports=0,
            word_size=256,
            write_size=8,
            num_words=32,
            read_latency_cycles=2,
            write_latency_cycles=3,
        )
    ],
    processing_elements=[
        ProcessingElementConfig(
            name="pe_0_0",
            mac_latency_cycles=2,
            dataflow_mode=PEDataflowMode.OUTPUT_STATIONARY,
            activation=PEOperandConfig(format=int3),
            weight=PEOperandConfig(format=int3),
            accumulator=acc,
        )
    ],
)

npu = NPU(config)
print(npu.stats())
```
