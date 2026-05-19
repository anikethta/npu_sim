import pytest

from npu_sim import (
    BiasConfig,
    CoreConfig,
    InterconnectConfig,
    InterconnectKind,
    LatencyConfig,
    NumericFormatConfig,
    NumericFormatKind,
    NPUConfig,
    PEDataflowMode,
    PEOperandConfig,
    ProcessingElementConfig,
    RequantizationConfig,
    SRAMScratchpadConfig,
    SystolicArrayConfig,
)


def test_default_config_exports_native_dict():
    config = NPUConfig()

    native = config.to_native_dict()

    assert native["cores"]["count"] == 1
    assert native["interconnect"]["kind"] == "mesh"
    assert native["latencies"]["noop"] == 1
    assert native["scratchpads"] == []
    assert native["processing_elements"] == []
    assert native["systolic_arrays"] == []


def test_config_validates_positive_core_count():
    with pytest.raises(ValueError, match="core count"):
        CoreConfig(count=0)


def test_config_accepts_string_interconnect_kind():
    config = InterconnectConfig(kind="ring")

    assert config.kind is InterconnectKind.RING


def test_latency_config_rejects_negative_cycles():
    with pytest.raises(ValueError, match="must be non-negative"):
        LatencyConfig({"dma": -1})


def test_sram_scratchpad_config_exports_native_shape():
    scratchpad = SRAMScratchpadConfig(
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

    native = NPUConfig(scratchpads=[scratchpad]).to_native_dict()

    assert scratchpad.capacity_bits == 8192
    assert native["scratchpads"] == [
        {
            "name": "local_sram",
            "num_rw_ports": 1,
            "num_r_ports": 0,
            "num_w_ports": 0,
            "word_size": 256,
            "write_size": 8,
            "num_words": 32,
            "read_latency_cycles": 2,
            "write_latency_cycles": 3,
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"name": "sram", "num_rw_ports": 0}, "at least one port"),
        ({"name": "sram", "word_size": 0}, "word_size"),
        ({"name": "sram", "write_size": 0}, "write_size"),
        ({"name": "sram", "num_words": 0}, "num_words"),
        ({"name": "sram", "write_size": 512}, "write_size"),
        ({"name": "sram", "word_size": 257}, "divisible"),
        ({"name": "sram", "read_latency_cycles": -1}, "read_latency"),
        ({"name": "sram", "write_latency_cycles": -1}, "write_latency"),
    ],
)
def test_sram_scratchpad_config_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SRAMScratchpadConfig(**kwargs)


def test_processing_element_config_exports_native_shape():
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    config = NPUConfig(
        processing_elements=[
            ProcessingElementConfig(
                name="pe_0_0",
                mac_latency_cycles=2,
                dataflow_mode=PEDataflowMode.WEIGHT_STATIONARY,
                activation=PEOperandConfig(format=int3),
                weight=PEOperandConfig(format=int3),
                accumulator=NumericFormatConfig(
                    kind=NumericFormatKind.SIGNED_INT, bits=32
                ),
            )
        ]
    )

    native = config.to_native_dict()

    assert native["processing_elements"] == [
        {
            "name": "pe_0_0",
            "mac_latency_cycles": 2,
            "dataflow_mode": "weight_stationary",
            "activation": {
                "format": {
                    "kind": "signed_int",
                    "bits": 3,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                }
            },
            "weight": {
                "format": {
                    "kind": "signed_int",
                    "bits": 3,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                }
            },
            "accumulator": {
                "kind": "signed_int",
                "bits": 32,
                "exponent_bits": 0,
                "mantissa_bits": 0,
            },
        }
    ]


def test_systolic_array_requantization_config_exports_native_shape():
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    int4 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=4)
    config = SystolicArrayConfig(
        name="array0",
        size=2,
        activation=PEOperandConfig(format=int3),
        weight=PEOperandConfig(format=int3),
        bias=BiasConfig(enabled=True, bias=[0, -1]),
        requantization=RequantizationConfig(
            enabled=True,
            target=int4,
            scale_multiplier=[3, 5],
            shift=[1, 2],
        ),
    )

    native_dict = config.to_native_dict()
    bias_native = native_dict["bias"]
    native = native_dict["requantization"]

    assert bias_native == {
        "enabled": True,
        "format": {
            "kind": "signed_int",
            "bits": 16,
            "exponent_bits": 0,
            "mantissa_bits": 0,
        },
        "bias": {"per_column": True, "values": [0, -1]},
        "placement": "after_systolic_array",
    }

    assert native == {
        "enabled": True,
        "target": {
            "kind": "signed_int",
            "bits": 4,
            "exponent_bits": 0,
            "mantissa_bits": 0,
        },
        "scale_multiplier": {"per_column": True, "values": [3, 5]},
        "shift": {"per_column": True, "values": [1, 2]},
        "rounding": True,
        "placement": "after_systolic_array",
    }


def test_systolic_array_requantization_validates_registers():
    with pytest.raises(ValueError, match="shift"):
        RequantizationConfig(shift=-1)

    with pytest.raises(ValueError, match="vector length"):
        SystolicArrayConfig(
            name="array0",
            size=2,
            requantization=RequantizationConfig(scale_multiplier=[1, 2, 3]),
        )


def test_processing_element_config_accepts_float_format_metadata():
    fp8 = NumericFormatConfig(
        kind=NumericFormatKind.FLOAT, bits=8, exponent_bits=4, mantissa_bits=3
    )

    assert fp8.to_native_dict() == {
        "kind": "float",
        "bits": 8,
        "exponent_bits": 4,
        "mantissa_bits": 3,
    }


def test_systolic_array_config_exports_native_shape():
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    config = NPUConfig(
        systolic_arrays=[
            SystolicArrayConfig(
                name="array0",
                size=2,
                mac_latency_cycles=1,
                dataflow_mode=PEDataflowMode.OUTPUT_STATIONARY,
                activation=PEOperandConfig(format=int3),
                weight=PEOperandConfig(format=int3),
                accumulator=NumericFormatConfig(
                    kind=NumericFormatKind.SIGNED_INT, bits=32
                ),
            )
        ]
    )

    native = config.to_native_dict()

    assert native["systolic_arrays"] == [
        {
            "name": "array0",
            "size": 2,
            "mac_latency_cycles": 1,
            "dataflow_mode": "output_stationary",
            "activation": {
                "format": {
                    "kind": "signed_int",
                    "bits": 3,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                }
            },
            "weight": {
                "format": {
                    "kind": "signed_int",
                    "bits": 3,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                }
            },
            "accumulator": {
                "kind": "signed_int",
                "bits": 32,
                "exponent_bits": 0,
                "mantissa_bits": 0,
            },
            "bias": {
                "enabled": False,
                "format": {
                    "kind": "signed_int",
                    "bits": 16,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                },
                "bias": {"per_column": False, "values": [0]},
                "placement": "after_systolic_array",
            },
            "requantization": {
                "enabled": False,
                "target": {
                    "kind": "signed_int",
                    "bits": 8,
                    "exponent_bits": 0,
                    "mantissa_bits": 0,
                },
                "scale_multiplier": {"per_column": False, "values": [1]},
                "shift": {"per_column": False, "values": [0]},
                "rounding": True,
                "placement": "after_systolic_array",
            },
        }
    ]


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"name": ""}, "name"),
        ({"name": "array", "size": 0}, "size"),
        ({"name": "array", "mac_latency_cycles": -1}, "mac_latency"),
    ],
)
def test_systolic_array_config_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        SystolicArrayConfig(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"kind": NumericFormatKind.SIGNED_INT, "bits": 0}, "bits"),
        (
            {"kind": NumericFormatKind.SIGNED_INT, "bits": 3, "exponent_bits": 1},
            "integer formats",
        ),
        ({"kind": NumericFormatKind.FLOAT, "bits": 8}, "exponent_bits"),
        (
            {
                "kind": NumericFormatKind.FLOAT,
                "bits": 8,
                "exponent_bits": 4,
                "mantissa_bits": 4,
            },
            "less than bits",
        ),
    ],
)
def test_numeric_format_config_validates_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        NumericFormatConfig(**kwargs)
