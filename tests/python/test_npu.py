import pytest

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
    int4 = NumericFormatConfig(kind=NumericFormatKind.SIGNED_INT, bits=4)
    fp8 = NumericFormatConfig(
        kind=NumericFormatKind.FLOAT, bits=8, exponent_bits=4, mantissa_bits=3
    )
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
                SystolicArrayConfig(
                    name="array_requant",
                    size=2,
                    mac_latency_cycles=1,
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
                ),
                SystolicArrayConfig(
                    name="array_fp8",
                    size=2,
                    activation=PEOperandConfig(format=int3),
                    weight=PEOperandConfig(format=int3),
                    accumulator=acc,
                    requantization=RequantizationConfig(
                        enabled=True,
                        target=fp8,
                        scale_multiplier=0.5,
                        shift=1,
                    ),
                ),
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

    log_path = npu.dump_systolic_array_log(result, tmp_path / "array.log")
    log = log_path.read_text(encoding="utf-8")
    assert "Systolic Array Cycle Log" in log
    assert "cycle 0:" in log
    assert "mac_start" in log
    assert "outputs:" in log

    ws_result = npu.run_systolic_array(
        "array_ws",
        activations=[[1, 2], [3, 1]],
        weights=[[2, 1], [1, -1]],
    )

    assert ws_result["outputs"] == [[4, -1], [7, 2]]
    assert ws_result["dataflow_mode"] == "weight_stationary"
    assert ws_result["current_cycle"] > result["current_cycle"]
    assert any(event["event"] == "weight_load" for event in ws_result["trace"])

    stream_result = npu.run_systolic_array_stream(
        "array_ws",
        activation_batches=[
            [[1, 2], [3, 1]],
            [[2, 0], [1, -1]],
        ],
        weight_batches=[
            [[2, 1], [1, -1]],
            [[1, 2], [3, 0]],
        ],
    )

    assert stream_result["output_batches"] == [
        [[4, -1], [7, 2]],
        [[2, 4], [-2, 2]],
    ]
    assert stream_result["outputs"] == [[2, 4], [-2, 2]]
    assert stream_result["batch_count"] == 2
    assert any(
        event["event"] == "weight_fifo_push" and event["batch"] == 1
        for event in stream_result["trace"]
    )
    assert any(
        event["event"] == "weight_fifo_pop" and event["batch"] == 1
        for event in stream_result["trace"]
    )
    assert any(
        event["event"] == "shadow_weight_load" and event["batch"] == 1
        for event in stream_result["trace"]
    )
    assert any(
        event["event"] == "shadow_weight_activate" and event["batch"] == 1
        for event in stream_result["trace"]
    )
    assert not any(
        event["event"].startswith("shadow_weight_fifo")
        for event in stream_result["trace"]
    )

    requant_result = npu.run_systolic_array(
        "array_requant",
        activations=[[1, 2], [3, 1]],
        weights=[[2, 1], [1, -1]],
    )

    assert requant_result["outputs"] == [[6, -2], [7, 1]]
    events = [event["event"] for event in requant_result["trace"]]
    for event_name in (
        "bias_add",
        "requant_multiply",
        "requant_round",
        "requant_shift",
        "requant_saturate",
    ):
        assert event_name in events
    assert any(
        event["event"] == "output_emit" and event["value"] == 7
        for event in requant_result["trace"]
    )
    pre_requant = next(
        event
        for event in requant_result["trace"]
        if event["event"] == "pre_requant_output"
        and event["row"] == 1
        and event["col"] == 0
    )
    final_emit = next(
        event
        for event in requant_result["trace"]
        if event["event"] == "output_emit"
        and event["row"] == 1
        and event["col"] == 0
    )
    assert pre_requant["value"] == 7
    assert final_emit["cycle"] == pre_requant["cycle"] + 5
    assert any(
        event["event"] == "requant_multiply"
        and event["requantization"]["scale_multiplier"] == 5
        and event["requantization"]["shift"] == 2
        for event in requant_result["trace"]
    )
    assert any(
        event["event"] == "bias_add"
        and event["bias"]["bias"] == -1
        and event["bias"]["format_bits"] == 16
        for event in requant_result["trace"]
    )

    html_path = npu.visualize_systolic_array(
        requant_result, tmp_path / "requant.html"
    )
    html = html_path.read_text(encoding="utf-8")
    assert "requant_multiply" in html
    assert "pre_requant_output" in html
    assert "Final Results" in html
    assert "M_o" in html
    assert "bias_fmt" in html
    assert "stage-item final" in html

    requant_log_path = npu.dump_systolic_array_log(
        requant_result, tmp_path / "requant.log"
    )
    requant_log = requant_log_path.read_text(encoding="utf-8")
    assert "requant_shift" in requant_log
    assert "bias_add" in requant_log
    assert "target=signed_int4" in requant_log

    requant_stream_result = npu.run_systolic_array_stream(
        "array_requant",
        activation_batches=[
            [[1, 2], [3, 1]],
            [[2, 0], [1, -1]],
        ],
        weight_batches=[
            [[2, 1], [1, -1]],
            [[1, 2], [3, 0]],
        ],
    )
    batch0_last_output = max(
        event["cycle"]
        for event in requant_stream_result["trace"]
        if event["event"] == "output_emit" and event["batch"] == 0
    )
    batch1_first_mac = min(
        event["cycle"]
        for event in requant_stream_result["trace"]
        if event["event"] == "mac_start" and event["batch"] == 1
    )
    assert batch1_first_mac < batch0_last_output

    fp8_result = npu.run_systolic_array(
        "array_fp8",
        activations=[[1, 2], [3, 1]],
        weights=[[2, 1], [1, -1]],
    )

    assert fp8_result["outputs"] == [[1.0, -0.25], [1.75, 0.5]]


def test_npu_run_advances_cycles():
    npu = NPU()

    stats = npu.run(cycles=8)

    assert stats.current_cycle == 8
    assert stats.event_count == 0
