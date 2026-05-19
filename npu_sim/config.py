"""Typed configuration objects for NPU simulator construction."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum


RegisterValue = int | float
RegisterValues = RegisterValue | Sequence[RegisterValue]


def _is_pipeline_sequence(value: object) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes))


def _validate_pipeline_register(
    value: object, name: str, *, integer_only: bool = False
) -> None:
    values = value if _is_pipeline_sequence(value) else (value,)
    for item in values:
        if isinstance(item, bool):
            raise ValueError(f"{name} register must not contain bool")
        if integer_only and not isinstance(item, int):
            raise ValueError(f"{name} register values must be integers")
        if not integer_only and not isinstance(item, (int, float)):
            raise ValueError(f"{name} register values must be numeric")
        if name == "shift" and item < 0:
            raise ValueError("requantization shift must be non-negative")


def _to_native_pipeline_register(value: RegisterValues) -> dict[str, object]:
    if _is_pipeline_sequence(value):
        return {"per_column": True, "values": list(value)}
    return {"per_column": False, "values": [value]}


class InterconnectKind(str, Enum):
    """Supported interconnect topology families for the initial scaffold."""

    BUS = "bus"
    MESH = "mesh"
    RING = "ring"


class NumericFormatKind(str, Enum):
    """Behavioral numeric format families supported by PE configs."""

    SIGNED_INT = "signed_int"
    UNSIGNED_INT = "unsigned_int"
    FLOAT = "float"


class PEDataflowMode(str, Enum):
    """Dataflow modes representable by PE configs."""

    OUTPUT_STATIONARY = "output_stationary"
    WEIGHT_STATIONARY = "weight_stationary"


class PipelinePlacement(str, Enum):
    """Attachment points for post-processing pipeline modules."""

    AFTER_SYSTOLIC_ARRAY = "after_systolic_array"


@dataclass(frozen=True)
class CoreConfig:
    """Configuration for a homogeneous group of compute cores."""

    count: int = 1
    frequency_hz: int = 1_000_000_000

    def __post_init__(self) -> None:
        if self.count <= 0:
            raise ValueError("core count must be positive")
        if self.frequency_hz <= 0:
            raise ValueError("core frequency_hz must be positive")


@dataclass(frozen=True)
class MemoryConfig:
    """High-level memory configuration placeholder."""

    capacity_bytes: int = 1 << 30
    bandwidth_bytes_per_cycle: int = 64

    def __post_init__(self) -> None:
        if self.capacity_bytes <= 0:
            raise ValueError("memory capacity_bytes must be positive")
        if self.bandwidth_bytes_per_cycle <= 0:
            raise ValueError("memory bandwidth_bytes_per_cycle must be positive")


@dataclass(frozen=True)
class InterconnectConfig:
    """Interconnect topology and width configuration."""

    kind: InterconnectKind = InterconnectKind.MESH
    link_width_bits: int = 256

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", InterconnectKind(self.kind))
        if self.link_width_bits <= 0:
            raise ValueError("interconnect link_width_bits must be positive")


@dataclass(frozen=True)
class LatencyConfig:
    """Named operation latencies, expressed in simulator cycles."""

    cycles: Mapping[str, int] = field(default_factory=lambda: {"noop": 1})

    def __post_init__(self) -> None:
        if not self.cycles:
            raise ValueError("latency table must contain at least one entry")
        for name, cycles in self.cycles.items():
            if not name:
                raise ValueError("latency names must be non-empty")
            if cycles < 0:
                raise ValueError(f"latency for {name!r} must be non-negative")


@dataclass(frozen=True)
class NumericFormatConfig:
    """Parameterized numeric format metadata for PE operands."""

    kind: NumericFormatKind = NumericFormatKind.SIGNED_INT
    bits: int = 8
    exponent_bits: int = 0
    mantissa_bits: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.kind, str):
            object.__setattr__(self, "kind", NumericFormatKind(self.kind))
        if self.bits <= 0 or self.bits > 63:
            raise ValueError("numeric format bits must be in [1, 63]")
        if self.kind is NumericFormatKind.FLOAT:
            if self.exponent_bits <= 0:
                raise ValueError("float exponent_bits must be positive")
            if self.mantissa_bits <= 0:
                raise ValueError("float mantissa_bits must be positive")
            if self.exponent_bits + self.mantissa_bits >= self.bits:
                raise ValueError(
                    "float exponent_bits + mantissa_bits must be less than bits"
                )
        elif self.exponent_bits != 0 or self.mantissa_bits != 0:
            raise ValueError("integer formats must not set float metadata")

    def to_native_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind.value,
            "bits": self.bits,
            "exponent_bits": self.exponent_bits,
            "mantissa_bits": self.mantissa_bits,
        }


@dataclass(frozen=True)
class PEOperandConfig:
    """PE operand format metadata."""

    format: NumericFormatConfig = field(default_factory=NumericFormatConfig)

    def to_native_dict(self) -> dict[str, object]:
        return {"format": self.format.to_native_dict()}


@dataclass(frozen=True)
class BiasConfig:
    """Register-like bias-adder pipeline configuration."""

    enabled: bool = False
    bias: RegisterValues = 0
    format: NumericFormatConfig = field(
        default_factory=lambda: NumericFormatConfig(
            kind=NumericFormatKind.SIGNED_INT, bits=16
        )
    )
    placement: PipelinePlacement | str = PipelinePlacement.AFTER_SYSTOLIC_ARRAY

    def __post_init__(self) -> None:
        if isinstance(self.placement, PipelinePlacement):
            object.__setattr__(self, "placement", self.placement.value)
        if not isinstance(self.placement, str) or not self.placement:
            raise ValueError("bias placement must be a non-empty string")
        _validate_pipeline_register(self.bias, "bias")
        if self.format.kind is not NumericFormatKind.FLOAT:
            _validate_pipeline_register(self.bias, "bias", integer_only=True)

    def validate_for_size(self, size: int) -> None:
        if _is_pipeline_sequence(self.bias) and len(self.bias) != size:
            raise ValueError("bias vector length must match systolic array size")

    def validate_precision(self, activation_bits: int, weight_bits: int) -> None:
        if not self.enabled or self.format.kind is NumericFormatKind.FLOAT:
            return
        if self.format.bits <= max(activation_bits, weight_bits):
            raise ValueError(
                "bias format bits must be greater than activation/weight bits"
            )

    def to_native_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "format": self.format.to_native_dict(),
            "bias": _to_native_pipeline_register(self.bias),
            "placement": self.placement,
        }


@dataclass(frozen=True)
class RequantizationConfig:
    """Register-like output requantization pipeline configuration."""

    enabled: bool = False
    target: NumericFormatConfig = field(default_factory=NumericFormatConfig)
    scale_multiplier: RegisterValues = 1
    shift: int | Sequence[int] = 0
    bias: RegisterValues | None = None
    rounding: bool = True
    placement: PipelinePlacement | str = PipelinePlacement.AFTER_SYSTOLIC_ARRAY

    def __post_init__(self) -> None:
        if isinstance(self.placement, PipelinePlacement):
            object.__setattr__(self, "placement", self.placement.value)
        if not isinstance(self.placement, str) or not self.placement:
            raise ValueError("requantization placement must be a non-empty string")
        _validate_pipeline_register(self.scale_multiplier, "scale_multiplier")
        _validate_pipeline_register(self.shift, "shift", integer_only=True)
        if self.bias is not None:
            _validate_pipeline_register(self.bias, "bias")

    def validate_for_size(self, size: int) -> None:
        """Validate per-column register lengths against an array size."""

        for name, value in (
            ("scale_multiplier", self.scale_multiplier),
            ("shift", self.shift),
            ("bias", self.bias),
        ):
            if value is not None and _is_pipeline_sequence(value) and len(value) != size:
                raise ValueError(
                    f"requantization {name} vector length must match systolic array size"
                )

    def to_native_dict(self) -> dict[str, object]:
        return {
            "enabled": self.enabled,
            "target": self.target.to_native_dict(),
            "scale_multiplier": _to_native_pipeline_register(
                self.scale_multiplier
            ),
            "shift": _to_native_pipeline_register(self.shift),
            "rounding": self.rounding,
            "placement": self.placement,
        }


@dataclass(frozen=True)
class ProcessingElementConfig:
    """Configuration for a SystemC processing element module."""

    name: str
    mac_latency_cycles: int = 1
    dataflow_mode: PEDataflowMode = PEDataflowMode.OUTPUT_STATIONARY
    activation: PEOperandConfig = field(default_factory=PEOperandConfig)
    weight: PEOperandConfig = field(default_factory=PEOperandConfig)
    accumulator: NumericFormatConfig = field(
        default_factory=lambda: NumericFormatConfig(
            kind=NumericFormatKind.SIGNED_INT, bits=32
        )
    )

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("processing element name must be non-empty")
        if self.mac_latency_cycles < 0:
            raise ValueError("processing element mac_latency_cycles must be non-negative")
        if isinstance(self.dataflow_mode, str):
            object.__setattr__(
                self, "dataflow_mode", PEDataflowMode(self.dataflow_mode)
            )

    def to_native_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "mac_latency_cycles": self.mac_latency_cycles,
            "dataflow_mode": self.dataflow_mode.value,
            "activation": self.activation.to_native_dict(),
            "weight": self.weight.to_native_dict(),
            "accumulator": self.accumulator.to_native_dict(),
        }


@dataclass(frozen=True)
class SystolicArrayConfig:
    """Configuration for a square SystemC systolic array module."""

    name: str
    size: int = 1
    mac_latency_cycles: int = 1
    dataflow_mode: PEDataflowMode = PEDataflowMode.OUTPUT_STATIONARY
    activation: PEOperandConfig = field(default_factory=PEOperandConfig)
    weight: PEOperandConfig = field(default_factory=PEOperandConfig)
    accumulator: NumericFormatConfig = field(
        default_factory=lambda: NumericFormatConfig(
            kind=NumericFormatKind.SIGNED_INT, bits=32
        )
    )
    bias: BiasConfig = field(default_factory=BiasConfig)
    requantization: RequantizationConfig = field(default_factory=RequantizationConfig)

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("systolic array name must be non-empty")
        if self.size <= 0:
            raise ValueError("systolic array size must be positive")
        if self.mac_latency_cycles < 0:
            raise ValueError("systolic array mac_latency_cycles must be non-negative")
        if isinstance(self.dataflow_mode, str):
            object.__setattr__(
                self, "dataflow_mode", PEDataflowMode(self.dataflow_mode)
            )
        if self.requantization.bias is not None and not self.bias.enabled:
            object.__setattr__(
                self,
                "bias",
                BiasConfig(
                    enabled=True,
                    bias=self.requantization.bias,
                    placement=self.requantization.placement,
                ),
            )
        self.bias.validate_for_size(self.size)
        self.bias.validate_precision(
            self.activation.format.bits, self.weight.format.bits
        )
        self.requantization.validate_for_size(self.size)

    def to_native_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "size": self.size,
            "mac_latency_cycles": self.mac_latency_cycles,
            "dataflow_mode": self.dataflow_mode.value,
            "activation": self.activation.to_native_dict(),
            "weight": self.weight.to_native_dict(),
            "accumulator": self.accumulator.to_native_dict(),
            "bias": self.bias.to_native_dict(),
            "requantization": self.requantization.to_native_dict(),
        }


@dataclass(frozen=True)
class SRAMScratchpadConfig:
    """Configuration for a timing/resource-mode SRAM scratchpad block."""

    name: str
    num_rw_ports: int = 1
    num_r_ports: int = 0
    num_w_ports: int = 0
    word_size: int = 256
    write_size: int = 8
    num_words: int = 32
    read_latency_cycles: int = 1
    write_latency_cycles: int = 1

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("scratchpad name must be non-empty")
        if self.num_rw_ports < 0:
            raise ValueError("scratchpad num_rw_ports must be non-negative")
        if self.num_r_ports < 0:
            raise ValueError("scratchpad num_r_ports must be non-negative")
        if self.num_w_ports < 0:
            raise ValueError("scratchpad num_w_ports must be non-negative")
        if self.num_rw_ports + self.num_r_ports + self.num_w_ports <= 0:
            raise ValueError("scratchpad must have at least one port")
        if self.word_size <= 0:
            raise ValueError("scratchpad word_size must be positive")
        if self.write_size <= 0:
            raise ValueError("scratchpad write_size must be positive")
        if self.num_words <= 0:
            raise ValueError("scratchpad num_words must be positive")
        if self.write_size > self.word_size:
            raise ValueError("scratchpad write_size must be <= word_size")
        if self.word_size % self.write_size != 0:
            raise ValueError("scratchpad word_size must be divisible by write_size")
        if self.read_latency_cycles < 0:
            raise ValueError("scratchpad read_latency_cycles must be non-negative")
        if self.write_latency_cycles < 0:
            raise ValueError("scratchpad write_latency_cycles must be non-negative")

    @property
    def capacity_bits(self) -> int:
        """Total scratchpad capacity in bits."""

        return self.word_size * self.num_words

    def to_native_dict(self) -> dict[str, object]:
        """Convert the scratchpad config to a binding-friendly dictionary."""

        return {
            "name": self.name,
            "num_rw_ports": self.num_rw_ports,
            "num_r_ports": self.num_r_ports,
            "num_w_ports": self.num_w_ports,
            "word_size": self.word_size,
            "write_size": self.write_size,
            "num_words": self.num_words,
            "read_latency_cycles": self.read_latency_cycles,
            "write_latency_cycles": self.write_latency_cycles,
        }


@dataclass(frozen=True)
class NPUConfig:
    """Top-level simulator configuration passed to the native backend."""

    cores: CoreConfig = field(default_factory=CoreConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    interconnect: InterconnectConfig = field(default_factory=InterconnectConfig)
    latencies: LatencyConfig = field(default_factory=LatencyConfig)
    scratchpads: Sequence[SRAMScratchpadConfig] = field(default_factory=tuple)
    processing_elements: Sequence[ProcessingElementConfig] = field(default_factory=tuple)
    systolic_arrays: Sequence[SystolicArrayConfig] = field(default_factory=tuple)

    def to_native_dict(self) -> dict[str, object]:
        """Convert the typed config to a simple binding-friendly dictionary."""

        return {
            "cores": {
                "count": self.cores.count,
                "frequency_hz": self.cores.frequency_hz,
            },
            "memory": {
                "capacity_bytes": self.memory.capacity_bytes,
                "bandwidth_bytes_per_cycle": self.memory.bandwidth_bytes_per_cycle,
            },
            "interconnect": {
                "kind": self.interconnect.kind.value,
                "link_width_bits": self.interconnect.link_width_bits,
            },
            "latencies": dict(self.latencies.cycles),
            "scratchpads": [
                scratchpad.to_native_dict() for scratchpad in self.scratchpads
            ],
            "processing_elements": [
                pe.to_native_dict() for pe in self.processing_elements
            ],
            "systolic_arrays": [
                array.to_native_dict() for array in self.systolic_arrays
            ],
        }
