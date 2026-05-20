# NPU Sim

A modular NPU simulation platform with an integrated Python frontend.

## Current Scaffold

- Python package: `npu_sim`
- Native backend: C++17, CMake, pybind11, SystemC
- Simulation model: SystemC-backed cycle-accurate simulation
- frontend Python API
- Implemented SystemC modules:
  - SRAM scratchpad with bit-level storage, configurable ports/shape/latency,
    four-state startup `X`, reads, writes, and timing counters
  - Processing element with configurable MAC latency, parameterized numeric
    formats, output-stationary/weight-stationary config, local accumulator, and
    C++-only behavioral MAC tests
  - Systolic array built from PE grids, with cycle-accurate matrix
    multiply, output-stationary and weight-stationary dataflows, batched
    streaming, unified FIFO ingress, and a nifty Python HTML visualization
- C++ layout:
  - `cpp/include/npu_sim/simulator.hpp`: Python-facing simulator facade
  - `cpp/include/npu_sim/backend/systemc/`: SystemC runtime/backend internals
  - `cpp/include/npu_sim/modules/`: reusable SystemC hardware modules

## Development Setup (MacOS)

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
- Public Python config can instantiate SRAMs, PEs, and systolic arrays.
- `NPU.run_systolic_array(...)` runs one square matrix multiply through an
  array. `NPU.run_systolic_array_stream(...)` streams multiple activation and
  weight matrix pairs through the same array.
- Systolic traces include batch IDs, FIFO push/pop events, PE arrivals,
  MAC starts/commits, optional post-output pipeline stages, output emissions,
  and weight-stationary shadow register activity.
- Systolic outputs can optionally run through independently configured
  post-output pipeline modules. `BiasConfig` models a higher-precision bias
  adder stage, while `RequantizationConfig` models fixed-point scale multiply,
  rounding, arithmetic shift, and target-format saturation. Integer targets
  clamp by parameterized bit width, while FP4/FP8-style targets use the current
  behavioral double model.
- Post-output pipeline configs include a `placement` string. The current
  backend attaches `after_systolic_array`; other placements are reserved for
  future modules such as vector units.
- Weight-stationary batching uses one logical weight FIFO stream. Later batches
  may be transferred from that FIFO into PE shadow registers while an earlier
  batch is computing; there is no separate shadow-weight FIFO.
- Weight-stationary streams may provide fewer weight batches than activation
  batches. The last loaded weight batch remains resident in the PEs and is
  reused until another weight batch is supplied.
- `NPU.visualize_systolic_array(...)` writes a self-contained HTML trace viewer
  with play/pause, stepping, speed control, PE-grid highlighting, FIFO/shadow
  activity, and per-batch outputs.

## Minimal Python Example

```python
from npu_sim import (
    BiasConfig,
    CoreConfig,
    NPU,
    NPUConfig,
    NumericFormatConfig,
    NumericFormatKind,
    PEDataflowMode,
    PEOperandConfig,
    ProcessingElementConfig,
    RequantizationConfig,
    SRAMScratchpadConfig,
    SystolicArrayConfig,
)

int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
acc = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=32)
int4 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=4)

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
    systolic_arrays=[
        SystolicArrayConfig(
            name="array0",
            size=2,
            mac_latency_cycles=1,
            dataflow_mode=PEDataflowMode.WEIGHT_STATIONARY,
            activation=PEOperandConfig(format=int3),
            weight=PEOperandConfig(format=int3),
            accumulator=acc,
            bias=BiasConfig(enabled=True, bias=[0, -1]),
            requantization=RequantizationConfig(
                enabled=True,
                target=int4,
                scale_multiplier=[3, 5],
                shift=[1, 2],
            ),
        )
    ],
)

npu = NPU(config)
print(npu.stats())

result = npu.run_systolic_array_stream(
    "array0",
    activation_batches=[
        [[1, 2], [3, 1]],
        [[2, 0], [1, -1]],
    ],
    weight_batches=[
        [[2, 1], [1, -1]],
        [[1, 2], [3, 0]],
    ],
)
print(result["output_batches"])
npu.visualize_systolic_array(result)
```
