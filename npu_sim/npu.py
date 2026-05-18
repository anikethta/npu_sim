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
    <label>Speed <input id="speed" type="range" min="100" max="1500" value="650"></label>
  </div>
  <div class="layout">
    <section class="panel">
      <div class="stream" id="streams"></div>
      <div class="grid" id="grid"></div>
    </section>
    <aside class="panel">
      <h2>Final Outputs</h2>
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

document.getElementById("summary").textContent =
  `array size ${{size}}x${{size}} | mode ${{result.dataflow_mode}} | operations ${{result.operation_count}} | final cycle ${{result.current_cycle}}`;

function renderMatrix(values) {{
  const host = document.getElementById("outputs");
  host.innerHTML = "";
  const matrix = document.createElement("div");
  matrix.className = "matrix";
  matrix.style.gridTemplateColumns = `repeat(${{size}}, 44px)`;
  values.forEach(row => row.forEach(value => {{
    const cell = document.createElement("div");
    cell.className = "cell";
    cell.textContent = value;
    matrix.appendChild(cell);
  }}));
  host.appendChild(matrix);
}}

function formatEvent(e) {{
  const where = e.row === null || e.col === null ? "" : ` PE(${{e.row}},${{e.col}})`;
  const kval = e.k === null ? "" : ` k=${{e.k}}`;
  const value = e.value === null || e.value === undefined ? "" : ` value=${{e.value}}`;
  return `${{e.event}}${{where}}${{kval}}${{value}}`;
}}

function render() {{
  cycleInput.value = String(cycle);
  cycleLabel.textContent = `${{cycle}} / ${{maxCycle}}`;
  document.querySelectorAll(".pe").forEach(pe => {{
    pe.className = "pe";
    pe.querySelector(".events").innerHTML = "";
  }});

  const events = trace.filter(e => e.cycle === cycle);
  const sideEvents = [];
  for (const event of events) {{
    if (event.row === null || event.col === null) {{
      sideEvents.push(formatEvent(event));
      continue;
    }}
    const pe = document.getElementById(`pe-${{event.row}}-${{event.col}}`);
    if (!pe) continue;
    if (event.event.includes("start") || event.event.includes("arrive")) pe.classList.add("active");
    if (event.event.includes("commit")) pe.classList.add("commit");
    if (event.event.includes("output")) pe.classList.add("output");
    const line = document.createElement("div");
    line.textContent = formatEvent(event);
    pe.querySelector(".events").appendChild(line);
  }}
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

renderMatrix(result.outputs || []);
render();
</script>
</body>
</html>
"""
        path.write_text(html, encoding="utf-8")
        return path

    def stats(self) -> SimulationStats:
        """Return current simulator counters."""

        return SimulationStats(**self._backend.stats())
