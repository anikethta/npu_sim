import pytest

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
    SystolicArrayConfig,
)


def test_npu_constructs_with_default_config():
    npu = NPU()

    assert npu.stats().current_cycle == 0
    assert npu.stats().component_count == 1


def test_npu_component_count_tracks_core_count():
    npu = NPU(NPUConfig(cores=CoreConfig(count=4)))

    assert npu.stats().component_count == 4


def test_npu_component_count_includes_scratchpads():
    npu = NPU(
        NPUConfig(
            cores=CoreConfig(count=4),
            scratchpads=[
                SRAMScratchpadConfig(name="local_sram"),
                SRAMScratchpadConfig(name="global_sram", num_r_ports=1),
            ],
        )
    )

    assert npu.stats().component_count == 6


def test_npu_component_count_includes_processing_elements():
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    acc = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=32)
    npu = NPU(
        NPUConfig(
            cores=CoreConfig(count=2),
            processing_elements=[
                ProcessingElementConfig(
                    name="pe_0_0",
                    mac_latency_cycles=2,
                    dataflow_mode=PEDataflowMode.OUTPUT_STATIONARY,
                    activation=PEOperandConfig(format=int3),
                    weight=PEOperandConfig(format=int3),
                    accumulator=acc,
                ),
                ProcessingElementConfig(
                    name="pe_0_1",
                    mac_latency_cycles=3,
                    dataflow_mode=PEDataflowMode.WEIGHT_STATIONARY,
                    activation=PEOperandConfig(format=int3),
                    weight=PEOperandConfig(format=int3),
                    accumulator=acc,
                ),
            ],
        )
    )

    assert npu.stats().component_count == 4


def systolic_config(name="array0", dataflow_mode=PEDataflowMode.OUTPUT_STATIONARY):
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    acc = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=32)
    return SystolicArrayConfig(
        name=name,
        size=2,
        mac_latency_cycles=1,
        dataflow_mode=dataflow_mode,
        activation=PEOperandConfig(format=int3),
        weight=PEOperandConfig(format=int3),
        accumulator=acc,
    )


def test_npu_component_count_includes_systolic_array_pes():
    npu = NPU(NPUConfig(cores=CoreConfig(count=2), systolic_arrays=[systolic_config()]))

    assert npu.stats().component_count == 6


def test_npu_runs_debug_pe_and_systolic_array_transactions(tmp_path):
    int3 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=3)
    acc = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=32)
    npu = NPU(
        NPUConfig(
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
                systolic_config("array_os"),
                systolic_config("array_ws", PEDataflowMode.WEIGHT_STATIONARY),
            ],
        )
    )

    with pytest.raises(ValueError):
        npu.run_systolic_array(
            "array_os", activations=[[1, 2]], weights=[[1, 2], [3, 4]]
        )

    with pytest.raises((ValueError, IndexError)):
        npu.run_systolic_array(
            "array_os",
            activations=[[4, 2], [3, 1]],
            weights=[[2, 1], [1, -1]],
        )

    with pytest.raises((ValueError, IndexError)):
        npu.mac("pe_0_0", activation=4, weight=1)

    mac_result = npu.mac("pe_0_0", activation=3, weight=-2)
    assert mac_result["value"] == -6
    assert mac_result["kind"] == "signed_int"
    assert mac_result["bits"] == 32
    assert mac_result["current_cycle"] == 2
    assert mac_result["mac_count"] == 1

    result = npu.run_systolic_array(
        "array_os",
        activations=[[1, 2], [3, 1]],
        weights=[[2, 1], [1, -1]],
    )

    assert result["outputs"] == [[4, -1], [7, 2]]
    assert result["operation_count"] == 8
    assert result["dataflow_mode"] == "output_stationary"
    assert result["current_cycle"] == 6

    inject_counts = {}
    for event in result["trace"]:
        if event["event"] == "inject_activation":
            inject_counts[event["cycle"]] = inject_counts.get(event["cycle"], 0) + 1
    assert [inject_counts[cycle] for cycle in sorted(inject_counts)] == [1, 2, 1]

    first_commit = min(
        event["cycle"] for event in result["trace"] if event["event"] == "mac_commit"
    )
    assert first_commit == 3

    output_path = npu.visualize_systolic_array(result, tmp_path / "array.html")
    html = output_path.read_text(encoding="utf-8")
    assert "Systolic Array Trace" in html
    assert "trace-data" in html
    assert "output_stationary" in html

    ws_result = npu.run_systolic_array(
        "array_ws",
        activations=[[1, 2], [3, 1]],
        weights=[[2, 1], [1, -1]],
    )

    assert ws_result["outputs"] == [[4, -1], [7, 2]]
    assert ws_result["dataflow_mode"] == "weight_stationary"
    assert ws_result["current_cycle"] > result["current_cycle"]
    assert any(event["event"] == "weight_load" for event in ws_result["trace"])


def test_npu_run_advances_cycles():
    npu = NPU()

    stats = npu.run(cycles=8)

    assert stats.current_cycle == 8
    assert stats.event_count == 0
