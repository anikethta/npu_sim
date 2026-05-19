"""Python facade for the native NPU simulator backend."""

from __future__ import annotations

from dataclasses import dataclass
from html import escape
import json
from pathlib import Path
from typing import Any

from npu_sim.config import NPUConfig


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
        trace_json = json.dumps(result, separators=(",", ":"))
        title = "Systolic Array Trace"
        html = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{escape(title)}</title>
<style>
:root {{
  color-scheme: light;
  font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  background: #f7f8fb;
  color: #17202a;
}}
body {{
  margin: 0;
  padding: 24px;
}}
main {{
  max-width: 1120px;
  margin: 0 auto;
}}
h1 {{
  font-size: 24px;
  margin: 0 0 16px;
}}
.toolbar, .summary {{
  display: flex;
  flex-wrap: wrap;
  align-items: center;
  gap: 10px;
  margin-bottom: 16px;
}}
button {{
  border: 1px solid #b8c2d2;
  border-radius: 6px;
  background: #ffffff;
  color: #17202a;
  min-width: 40px;
  min-height: 34px;
  cursor: pointer;
}}
button:hover {{
  background: #edf2f7;
}}
input[type="range"] {{
  width: 280px;
}}
.layout {{
  display: grid;
  grid-template-columns: minmax(300px, 1fr) 320px;
  gap: 18px;
}}
.grid {{
  display: grid;
  gap: 8px;
  align-items: stretch;
}}
.requant-pipeline {{
  margin-top: 12px;
  border-top: 1px solid #cbd5e1;
  padding-top: 12px;
}}
.pipeline-title {{
  font-size: 13px;
  font-weight: 700;
  margin-bottom: 8px;
}}
.pipeline-stages {{
  display: grid;
  grid-template-columns: repeat(5, minmax(96px, 1fr));
  gap: 8px;
}}
.pipeline-stage {{
  min-height: 82px;
  border: 1px solid #b8c2d2;
  border-radius: 8px;
  background: #fbfdff;
  padding: 8px;
}}
.pipeline-stage.active {{
  border-color: #7c3aed;
  box-shadow: 0 0 0 2px rgba(124, 58, 237, 0.18);
}}
.pipeline-stage.final.active {{
  border-color: #d97706;
  box-shadow: 0 0 0 2px rgba(217, 119, 6, 0.22);
}}
.stage-name {{
  font-size: 12px;
  font-weight: 700;
  margin-bottom: 6px;
}}
.stage-items {{
  display: grid;
  gap: 4px;
}}
.stage-item {{
  border: 1px solid #ddd6fe;
  border-radius: 6px;
  background: #f5f3ff;
  color: #312e81;
  padding: 4px 5px;
  font-size: 11px;
  line-height: 1.25;
  overflow-wrap: anywhere;
}}
.stage-item.final {{
  border-color: #f59e0b;
  background: #fffbeb;
  color: #78350f;
}}
.stage-empty {{
  color: #64748b;
  font-size: 12px;
}}
.pe {{
  min-height: 112px;
  border: 1px solid #b8c2d2;
  border-radius: 8px;
  background: #ffffff;
  padding: 8px;
  display: grid;
  grid-template-rows: auto 1fr;
  box-shadow: 0 1px 2px rgba(16, 24, 40, 0.06);
}}
.pe.active {{
  outline: 3px solid #2563eb;
}}
.pe.commit {{
  outline: 3px solid #16a34a;
}}
.pe.output {{
  outline: 3px solid #d97706;
}}
.pe.requant {{
  outline: 3px solid #7c3aed;
}}
.pe.quant-output {{
  outline: 3px solid #f59e0b;
}}
.pe.shadow {{
  outline: 3px solid #9333ea;
}}
.pe.fifo {{
  outline: 3px solid #0891b2;
}}
.coord {{
  font-weight: 700;
  font-size: 13px;
}}
.events {{
  font-size: 12px;
  line-height: 1.35;
  overflow-wrap: anywhere;
}}
.panel {{
  border: 1px solid #b8c2d2;
  border-radius: 8px;
  background: #ffffff;
  padding: 12px;
}}
.panel h2 {{
  margin: 0 0 8px;
  font-size: 15px;
}}
.matrix {{
  display: grid;
  gap: 6px;
  width: max-content;
}}
.cell {{
  border: 1px solid #cbd5e1;
  border-radius: 6px;
  min-width: 44px;
  min-height: 32px;
  display: grid;
  place-items: center;
  background: #f8fafc;
  font-variant-numeric: tabular-nums;
}}
.stream {{
  font-size: 13px;
  margin-bottom: 8px;
}}
@media (max-width: 760px) {{
  body {{ padding: 12px; }}
  .layout {{ grid-template-columns: 1fr; }}
  .pipeline-stages {{ grid-template-columns: 1fr; }}
  input[type="range"] {{ width: 100%; }}
}}
</style>
</head>
<body>
<main>
  <h1>Systolic Array Trace</h1>
  <div class="summary" id="summary"></div>
  <div class="toolbar">
    <button id="prev" title="Previous cycle">Prev</button>
    <button id="play" title="Play or pause">Play</button>
    <button id="next" title="Next cycle">Next</button>
    <label>Cycle <input id="cycle" type="range" min="0" value="0"></label>
    <span id="cycleLabel"></span>
    <label>Speed <input id="speed" type="range" min="100" max="2000" value="1000"></label>
  </div>
  <div class="layout">
    <section class="panel">
      <div class="stream" id="streams"></div>
      <div class="grid" id="grid"></div>
      <div class="requant-pipeline">
        <div class="pipeline-title">Requantization Pipeline</div>
        <div class="pipeline-stages" id="requantPipeline"></div>
      </div>
    </section>
    <aside class="panel">
      <h2>Final Results</h2>
      <div id="outputs"></div>
      <h2 style="margin-top:16px;">Cycle Events</h2>
      <div id="eventList"></div>
    </aside>
  </div>
</main>
<script id="trace-data" type="application/json">{trace_json}</script>
<script>
const result = JSON.parse(document.getElementById("trace-data").textContent);
const size = result.size;
const trace = result.trace || [];
const maxCycle = Math.max(result.current_cycle || 0, ...trace.map(e => e.cycle || 0));
let cycle = 0;
let timer = null;

const grid = document.getElementById("grid");
grid.style.gridTemplateColumns = `repeat(${{size}}, minmax(92px, 1fr))`;
for (let r = 0; r < size; r++) {{
  for (let c = 0; c < size; c++) {{
    const pe = document.createElement("div");
    pe.className = "pe";
    pe.id = `pe-${{r}}-${{c}}`;
    pe.innerHTML = `<div class="coord">PE(${{r}}, ${{c}})</div><div class="events"></div>`;
    grid.appendChild(pe);
  }}
}}

const cycleInput = document.getElementById("cycle");
cycleInput.max = String(maxCycle);
const cycleLabel = document.getElementById("cycleLabel");
const eventList = document.getElementById("eventList");
const streams = document.getElementById("streams");
const requantPipeline = document.getElementById("requantPipeline");
const requantStages = [
  ["bias_add", "Bias"],
  ["requant_multiply", "Multiply"],
  ["requant_round", "Round"],
  ["requant_shift", "Shift"],
  ["requant_saturate", "Saturate"],
];

for (const [eventName, label] of requantStages) {{
  const stage = document.createElement("div");
  stage.className = eventName === "requant_saturate" ? "pipeline-stage final" : "pipeline-stage";
  stage.id = `stage-${{eventName}}`;
  stage.innerHTML = `<div class="stage-name">${{label}}</div><div class="stage-items"></div>`;
  requantPipeline.appendChild(stage);
}}

document.getElementById("summary").textContent =
  `array size ${{size}}x${{size}} | mode ${{result.dataflow_mode}} | batches ${{result.batch_count || 1}} | operations ${{result.operation_count}} | final cycle ${{result.current_cycle}}`;

function renderSingleMatrix(values, label) {{
  const section = document.createElement("div");
  const heading = document.createElement("h3");
  heading.textContent = label;
  heading.style.fontSize = "13px";
  heading.style.margin = "10px 0 6px";
  section.appendChild(heading);
  const matrix = document.createElement("div");
  matrix.className = "matrix";
  matrix.style.gridTemplateColumns = `repeat(${{size}}, 44px)`;
  values.forEach(row => row.forEach(value => {{
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.textContent = value;
    matrix.appendChild(cell);
  }}));
  section.appendChild(matrix);
  return section;
}}

function renderMatrix(values, batches) {{
  const host = document.getElementById("outputs");
  host.innerHTML = "";
  if (batches && batches.length > 1) {{
    batches.forEach((matrix, index) => {{
      host.appendChild(renderSingleMatrix(matrix, `Batch ${{index}}`));
    }});
    return;
  }}
  host.appendChild(renderSingleMatrix(values, "Outputs"));
}}

function formatEvent(e) {{
  const where = e.row === null || e.col === null ? "" : ` PE(${{e.row}},${{e.col}})`;
  const kval = e.k === null ? "" : ` k=${{e.k}}`;
  const batch = e.batch === null || e.batch === undefined ? "" : ` batch=${{e.batch}}`;
  const value = e.value === null || e.value === undefined ? "" : ` value=${{e.value}}`;
  let requant = "";
  if (e.requantization) {{
    const r = e.requantization;
    requant = ` M_o=${{r.scale_multiplier}} shift=${{r.shift}} target=${{r.target_kind}}${{r.target_bits}}`;
  }}
  let bias = "";
  if (e.bias) {{
    bias = ` bias=${{e.bias.bias}} bias_fmt=${{e.bias.format_kind}}${{e.bias.format_bits}}`;
  }}
  return `${{e.event}}${{batch}}${{where}}${{kval}}${{value}}${{bias}}${{requant}}`;
}}

function formatPipelineItem(e) {{
  const row = e.row === null ? "?" : e.row;
  const col = e.col === null ? "?" : e.col;
  const batch = e.batch === null || e.batch === undefined ? "?" : e.batch;
  const value = e.value === null || e.value === undefined ? "X" : e.value;
  let detail = "";
  if (e.bias) {{
    detail = ` b=${{e.bias.bias}} fmt=${{e.bias.format_kind}}${{e.bias.format_bits}}`;
  }}
  if (e.requantization) {{
    const r = e.requantization;
    detail = ` M=${{r.scale_multiplier}} s=${{r.shift}}`;
  }}
  return `B${{batch}} PE(${{row}},${{col}}) v=${{value}}${{detail}}`;
}}

function render() {{
  cycleInput.value = String(cycle);
  cycleLabel.textContent = `${{cycle}} / ${{maxCycle}}`;
  document.querySelectorAll(".pe").forEach(pe => {{
    pe.className = "pe";
    pe.querySelector(".events").innerHTML = "";
  }});
  document.querySelectorAll(".pipeline-stage").forEach(stage => {{
    stage.className = stage.id === "stage-requant_saturate" ? "pipeline-stage final" : "pipeline-stage";
    stage.querySelector(".stage-items").innerHTML = "";
  }});

  const events = trace.filter(e => e.cycle === cycle);
  const sideEvents = [];
  for (const event of events) {{
    if (event.event.startsWith("requant_") || event.event === "bias_add") {{
      const stage = document.getElementById(`stage-${{event.event}}`);
      if (stage) {{
        stage.classList.add("active");
        const item = document.createElement("div");
        item.className = event.event === "requant_saturate" ? "stage-item final" : "stage-item";
        item.textContent = formatPipelineItem(event);
        stage.querySelector(".stage-items").appendChild(item);
      }}
      continue;
    }}
    if (
      event.event === "output_emit" &&
      events.some(e =>
        e.event === "requant_saturate" &&
        e.batch === event.batch &&
        e.row === event.row &&
        e.col === event.col
      )
    ) {{
      continue;
    }}
    if (event.row === null || event.col === null) {{
      sideEvents.push(formatEvent(event));
      continue;
    }}
    const pe = document.getElementById(`pe-${{event.row}}-${{event.col}}`);
    if (!pe) continue;
    if (event.event.includes("start") || event.event.includes("arrive")) pe.classList.add("active");
    if (event.event.includes("commit")) pe.classList.add("commit");
    if (event.event.includes("output")) pe.classList.add("output");
    if (event.event.includes("shadow")) pe.classList.add("shadow");
    if (event.event.includes("fifo")) pe.classList.add("fifo");
    const line = document.createElement("div");
    line.textContent = formatEvent(event);
    pe.querySelector(".events").appendChild(line);
  }}
  document.querySelectorAll(".pipeline-stage").forEach(stage => {{
    const items = stage.querySelector(".stage-items");
    if (!items.children.length) {{
      const empty = document.createElement("div");
      empty.className = "stage-empty";
      empty.textContent = "Idle";
      items.appendChild(empty);
    }}
  }});
  streams.textContent = sideEvents.length ? sideEvents.join(" | ") : "No side injections this cycle";
  eventList.innerHTML = events.length
    ? events.map(e => `<div>${{formatEvent(e)}}</div>`).join("")
    : "<div>No events</div>";
}}

document.getElementById("prev").onclick = () => {{ cycle = Math.max(0, cycle - 1); render(); }};
document.getElementById("next").onclick = () => {{ cycle = Math.min(maxCycle, cycle + 1); render(); }};
cycleInput.oninput = () => {{ cycle = Number(cycleInput.value); render(); }};
document.getElementById("play").onclick = () => {{
  const play = document.getElementById("play");
  if (timer) {{
    clearInterval(timer);
    timer = null;
    play.textContent = "Play";
    return;
  }}
  play.textContent = "Pause";
  timer = setInterval(() => {{
    cycle = cycle >= maxCycle ? 0 : cycle + 1;
    render();
  }}, Number(document.getElementById("speed").value));
}};

renderMatrix(result.outputs || [], result.output_batches || []);
render();
</script>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path

    def dump_systolic_array_log(
        self,
        result: dict[str, Any],
        output_path: str | Path = "systolic_array.log",
    ) -> Path:
        """Write a cycle-by-cycle text log for a systolic trace."""

        path = Path(output_path)
        trace = list(result.get("trace", []))
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

    def stats(self) -> SimulationStats:
        """Return current simulator counters."""

        return SimulationStats(**self._backend.stats())
