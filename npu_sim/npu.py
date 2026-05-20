"""Python facade for the native NPU simulator backend."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
from importlib.resources import files
import json
from pathlib import Path
from typing import Any

from npu_sim.config import NPUConfig


_TEMPLATE_PACKAGE = "npu_sim.templates"


def _render_template(template_name: str, replacements: dict[str, str]) -> str:
    """Render a bundled HTML template with literal sentinel replacements."""

    template = (
        files(_TEMPLATE_PACKAGE).joinpath(template_name).read_text(encoding="utf-8")
    )
    for placeholder, value in replacements.items():
        template = template.replace(placeholder, value)
    return template


@dataclass(frozen=True)
class SimulationStats:
    """Small Python representation of backend simulation counters."""

    current_cycle: int
    event_count: int
    component_count: int


class NPU:
    """User-facing NPU simulator object."""

    def __init__(self, config: NPUConfig | None = None) -> None:
        self.config = config or NPUConfig()

        from npu_sim import _native

        self._backend = _native.Simulator(self.config.to_native_dict())

    def run(self, cycles: int | None = None) -> SimulationStats:
        """Run pending simulation work, optionally bounded by cycle count."""

        if cycles is None:
            stats = self._backend.run()
        else:
            if cycles < 0:
                raise ValueError("cycles must be non-negative")
            stats = self._backend.run_until(self.stats().current_cycle + cycles)
        return SimulationStats(**stats)

    def mac(
        self, pe_name: str, activation: int | float, weight: int | float
    ) -> dict[str, object]:
        """Run one debug MAC transaction on a named processing element."""

        return dict(self._backend.mac(pe_name, activation, weight))

    def run_systolic_array(
        self,
        name: str,
        activations: list[list[int | float]],
        weights: list[list[int | float]],
        trace: bool = True,
    ) -> dict[str, Any]:
        """Run a debug matrix multiply transaction on a named systolic array."""

        return dict(
            self._backend.run_systolic_array(name, activations, weights, trace)
        )

    def run_systolic_array_stream(
        self,
        name: str,
        activation_batches: list[list[list[int | float]]],
        weight_batches: list[list[list[int | float]]],
        trace: bool = True,
    ) -> dict[str, Any]:
        """Run multiple matrix multiply transactions through one array."""

        return dict(
            self._backend.run_systolic_array_stream(
                name, activation_batches, weight_batches, trace
            )
        )

    def visualize_systolic_array(
        self,
        result: dict[str, Any],
        output_path: str | Path = "systolic_array.html",
    ) -> Path:
        """Write a self-contained HTML visualization for a systolic trace."""

        path = Path(output_path)
        systolic_result = dict(result)
        systolic_result["trace"] = [
            event
            for event in result.get("trace", [])
            if not str(event.get("event", "")).startswith("vpu_")
            and not (
                str(event.get("event", "")).startswith("requant_")
                and (event.get("requantization") or {}).get("placement")
                == "after_vpu"
            )
        ]
        pipeline_stages = []
        if any(
            event.get("event") == "bias_add"
            for event in systolic_result["trace"]
        ):
            pipeline_stages.append(["bias_add", "Bias"])
        pipeline_stages.append(["spu_output_fifo_push", "SPU FIFO"])
        if any(
            str(event.get("event", "")).startswith("requant_")
            and (event.get("requantization") or {}).get("placement")
            == "after_systolic_array"
            for event in systolic_result["trace"]
        ):
            pipeline_stages.extend(
                [
                    ["requant_multiply", "Multiply"],
                    ["requant_round", "Round"],
                    ["requant_shift", "Shift"],
                    ["requant_saturate", "Saturate"],
                ]
            )

        title = "Systolic Array Trace"
        html = _render_template(
            "systolic_array.html",
            {
                "__TITLE__": escape(title),
                "__TRACE_JSON__": json.dumps(systolic_result, separators=(",", ":")),
                "__PIPELINE_STAGES_JSON__": json.dumps(
                    pipeline_stages, separators=(",", ":")
                ),
            },
        )
        path.write_text(html, encoding="utf-8")
        return path

    def dump_systolic_array_log(
        self,
        result: dict[str, Any],
        output_path: str | Path = "systolic_array.log",
    ) -> Path:
        """Write a cycle-by-cycle text log for a systolic trace."""

        path = Path(output_path)
        trace = [
            event
            for event in result.get("trace", [])
            if not str(event.get("event", "")).startswith("vpu_")
            and not (
                str(event.get("event", "")).startswith("requant_")
                and (event.get("requantization") or {}).get("placement")
                == "after_vpu"
            )
        ]
        events_by_cycle: dict[int, list[dict[str, Any]]] = {}
        for event in trace:
            cycle = int(event.get("cycle", 0))
            events_by_cycle.setdefault(cycle, []).append(event)

        max_cycle = int(result.get("current_cycle", 0))
        if events_by_cycle:
            max_cycle = max(max_cycle, max(events_by_cycle))

        def format_event(event: dict[str, Any]) -> str:
            parts = [str(event.get("event", "event"))]
            if event.get("batch") is not None:
                parts.append(f"batch={event['batch']}")
            if event.get("row") is not None and event.get("col") is not None:
                parts.append(f"pe=({event['row']},{event['col']})")
            elif event.get("row") is not None:
                parts.append(f"row={event['row']}")
            elif event.get("col") is not None:
                parts.append(f"col={event['col']}")
            if event.get("k") is not None:
                parts.append(f"k={event['k']}")
            if event.get("value") is not None:
                parts.append(f"value={event['value']}")
            requantization = event.get("requantization")
            if requantization:
                parts.append(f"M_o={requantization.get('scale_multiplier')}")
                parts.append(f"shift={requantization.get('shift')}")
                parts.append(
                    "target="
                    f"{requantization.get('target_kind')}"
                    f"{requantization.get('target_bits')}"
                )
            bias = event.get("bias")
            if bias:
                parts.append(f"bias={bias.get('bias')}")
                parts.append(
                    f"bias_format={bias.get('format_kind')}{bias.get('format_bits')}"
                )
            lanes = event.get("lanes")
            if lanes:
                parts.append(f"dequeue_asserted={event.get('dequeue_asserted')}")
                lane_parts = []
                for index, lane in enumerate(lanes):
                    if lane.get("valid"):
                        lane_parts.append(
                            f"L{index}=({lane.get('row')},{lane.get('col')}):"
                            f"{lane.get('value')}"
                        )
                    else:
                        lane_parts.append(f"L{index}=0/void")
                parts.append("lanes=[" + " ".join(lane_parts) + "]")
            return " ".join(parts)

        lines = [
            "Systolic Array Cycle Log",
            f"array_size={result.get('size')}",
            f"dataflow_mode={result.get('dataflow_mode')}",
            f"batch_count={result.get('batch_count', 1)}",
            f"operation_count={result.get('operation_count')}",
            f"final_cycle={result.get('current_cycle')}",
            "",
        ]

        for cycle in range(max_cycle + 1):
            events = events_by_cycle.get(cycle, [])
            lines.append(f"cycle {cycle}:")
            if not events:
                lines.append("  idle")
            else:
                for event in events:
                    lines.append(f"  {format_event(event)}")
            lines.append("")

        output_batches = result.get("output_batches")
        if output_batches and len(output_batches) > 1:
            lines.append("output_batches:")
            for batch_index, matrix in enumerate(output_batches):
                lines.append(f"  batch {batch_index}:")
                for row in matrix:
                    lines.append(f"    {row}")
        else:
            lines.append("outputs:")
            for row in result.get("outputs", []):
                lines.append(f"  {row}")

        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def visualize_vpu(
        self,
        result: dict[str, Any],
        output_path: str | Path = "vpu.html",
    ) -> Path:
        """Write a self-contained HTML visualization for VPU trace events."""

        path = Path(output_path)
        vpu_events = [
            event
            for event in result.get("trace", [])
            if str(event.get("event", "")).startswith("vpu_")
            or (
                str(event.get("event", "")).startswith("requant_")
                and (event.get("requantization") or {}).get("placement")
                == "after_vpu"
            )
        ]
        vpu_result = dict(result)
        vpu_result["trace"] = vpu_events

        title = "VPU Trace"
        html = _render_template(
            "vpu.html",
            {
                "__TITLE__": escape(title),
                "__TRACE_JSON__": json.dumps(vpu_result, separators=(",", ":")),
            },
        )
        path.write_text(html, encoding="utf-8")
        return path

    def dump_vpu_log(
        self,
        result: dict[str, Any],
        output_path: str | Path = "vpu.log",
    ) -> Path:
        """Write a cycle-by-cycle text log for VPU trace events."""

        path = Path(output_path)
        trace = [
            event
            for event in result.get("trace", [])
            if str(event.get("event", "")).startswith("vpu_")
            or (
                str(event.get("event", "")).startswith("requant_")
                and (event.get("requantization") or {}).get("placement")
                == "after_vpu"
            )
        ]
        events_by_cycle: dict[int, list[dict[str, Any]]] = {}
        for event in trace:
            events_by_cycle.setdefault(int(event.get("cycle", 0)), []).append(event)

        max_cycle = int(result.get("current_cycle", 0))
        if events_by_cycle:
            max_cycle = max(max_cycle, max(events_by_cycle))

        def format_event(event: dict[str, Any]) -> str:
            parts = [str(event.get("event", "event"))]
            if event.get("batch") is not None:
                parts.append(f"batch={event['batch']}")
            if event.get("row") is not None and event.get("col") is not None:
                parts.append(f"pe=({event['row']},{event['col']})")
            if event.get("value") is not None:
                parts.append(f"value={event['value']}")
            requantization = event.get("requantization")
            if requantization:
                parts.append(f"M_o={requantization.get('scale_multiplier')}")
                parts.append(f"shift={requantization.get('shift')}")
                parts.append(f"placement={requantization.get('placement')}")
                parts.append(
                    "target="
                    f"{requantization.get('target_kind')}"
                    f"{requantization.get('target_bits')}"
                )
            return " ".join(parts)

        lines = [
            "VPU Cycle Log",
            f"array_size={result.get('size')}",
            f"batch_count={result.get('batch_count', 1)}",
            f"final_cycle={result.get('current_cycle')}",
            "",
        ]
        for cycle in range(max_cycle + 1):
            lines.append(f"cycle {cycle}:")
            events = events_by_cycle.get(cycle, [])
            if not events:
                lines.append("  idle")
            else:
                for event in events:
                    lines.append(f"  {format_event(event)}")
            lines.append("")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return path

    def stats(self) -> SimulationStats:
        """Return current simulator counters."""

        return SimulationStats(**self._backend.stats())
