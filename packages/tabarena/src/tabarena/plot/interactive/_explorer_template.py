"""HTML/CSS/JS template for the self-contained interactive Pareto explorer.

Kept as a Python string constant (rather than a package-data file) so it ships
with the package without any build-system data-file configuration. The
placeholders (``__BASE_CSS__``, ``__BASE_JS__``, ``__PAGE_TITLE__``,
``__CONFIG_JSON__``, ``__POINTS_JSON__``) are substituted by
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`; the
chrome and helpers spliced in via the first two are shared with the other
explorers (see :mod:`tabarena.plot.interactive._explorer_shared`).

The rendered page has zero external dependencies (no plotting library, fonts,
or CDN assets), renders in light and dark mode via ``prefers-color-scheme``,
and offers: per-method highlight chips grouped by model family, family-level
toggle buttons, a metric selector (e.g. Improvability vs Elo) and a time-axis
selector (e.g. train, inference or total time; per 1K samples or per task;
median or mean) that re-anchor the Pareto front, a subset selector when the
page holds several task subsets (each point names its ``subset``), hover
tooltips with exact values, dashed-ring marking of partially imputed methods,
typed axis limits kept per view (subset, metric and time axis), an editable
figure title drawn above the chart (by default the page's title and subset),
methods renamed from their chips, and a collapsible data table.
"""

from __future__ import annotations

EXPLORER_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__
  /* the selectors' labels stay on one line beside long option texts */
  .metricpick { white-space: nowrap; }
  .metricpick select { max-width: 100%; }

  /* Two-column layout: controls + chips in a side panel, chart beside it.
     ``chips-right`` mirrors the columns. Wraps to stacked when narrow. */
  .explorer-grid { display: flex; gap: 18px; align-items: flex-start; }
  .explorer-grid.chips-right { flex-direction: row-reverse; }
  .sidebox { flex: 0 0 330px; min-width: 250px; display: flex; flex-direction: column; gap: 10px; }
  .mainbox { flex: 1 1 auto; min-width: 0; }
  @media (max-width: 860px) {
    .explorer-grid { flex-wrap: wrap; }
    .sidebox { flex: 1 1 100%; }
  }

  .legendstrip .legendbreak { flex-basis: 100%; height: 0; }
  .chartbox { position: relative; }
  /* The chart is sized in device pixels by render() rather than scaled from a
     viewBox: scaling stretched the type along with the plot, and a viewBox tall
     enough to read made the whole panel own the screen. */
  .chartbox svg { display: block; }
  /* Chips scroll within the column so the panel height follows the chart. */
  .sidebox .chips { overflow-y: auto; scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent; }
  .sidebox .chips::-webkit-scrollbar { width: 9px; }
  .sidebox .chips::-webkit-scrollbar-thumb {
    background: var(--pt-muted); border-radius: 8px; border: 3px solid transparent; background-clip: content-box;
  }

  /* Typed axis limits: one row per axis, min and max side by side. An empty box shows the
     automatic limit as its placeholder; a value the axis cannot use is outlined dashed. */
  .limits {
    display: grid; grid-template-columns: auto minmax(0, 1fr) minmax(0, 1fr); align-items: center;
    gap: 5px 7px; width: 100%; font-size: 12.5px; font-weight: 600; color: var(--muted);
  }
  .limits .limits-head { font-size: 11.5px; font-weight: 600; }
  .limits input {
    width: 100%; box-sizing: border-box; min-width: 0;
    font: 600 12.5px/1.2 system-ui, sans-serif; color: var(--ink);
    background: var(--chip-bg); border: 1px solid var(--line); border-radius: 7px; padding: 5px 7px;
  }
  .limits input::placeholder { color: var(--muted); font-weight: 500; }
  .limits input[aria-invalid="true"] { border: 1.5px dashed var(--muted); }
  .limits input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .limits .btnrow { grid-column: 1 / -1; }

  /* The figure title box spans the panel. */
  .titlepick { width: 100%; }
  .titlepick input {
    flex: 1 1 auto; min-width: 0; box-sizing: border-box;
    font: 600 12.5px/1.2 system-ui, sans-serif; color: var(--ink);
    background: var(--chip-bg); border: 1px solid var(--line); border-radius: 7px; padding: 5px 7px;
  }
  .titlepick input::placeholder { color: var(--muted); font-weight: 500; }
  .titlepick input:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }

  /* A method chip's rename button sits on the chip's corner while the chip is hovered or focused,
     so the chips keep their layout; the rename box takes the chip's place while it is open. */
  .chipwrap { position: relative; display: inline-flex; }
  .chip-edit {
    position: absolute; top: -7px; right: -6px; width: 18px; height: 18px; padding: 0;
    display: none; align-items: center; justify-content: center;
    font: 600 11px/1 system-ui, sans-serif; color: var(--ink); cursor: pointer;
    background: var(--card); border: 1px solid var(--line); border-radius: 50%;
  }
  .chipwrap:hover .chip-edit, .chipwrap:focus-within .chip-edit { display: inline-flex; }
  .chip-edit:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  .chip-rename {
    font: 500 12.5px/1 system-ui, sans-serif; color: var(--ink); background: var(--card);
    border: 1px solid var(--accent); border-radius: 999px; padding: 4px 9px; min-width: 9ch; max-width: 100%;
  }
  .chip-rename:focus-visible { outline: none; }
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, chart and legend only — for slides and papers">Paper view</button>
  </div>
  <div class="explorer-grid" id="grid">
    <div class="sidebox">
      <div class="controls">
        <label class="metricpick titlepick">Title
          <input id="fig-title" type="text" spellcheck="false" placeholder="No title" aria-label="Figure title">
        </label>
        <label class="metricpick" id="subsetpick" hidden>Subset
          <select id="subset-select"></select>
        </label>
        <label class="metricpick" id="metricpick" hidden>Y-axis
          <select id="metric-select"></select>
        </label>
        <label class="metricpick" id="xaxispick" hidden>X-axis
          <select id="xaxis-select"></select>
        </label>
        <div class="btnrow">
          <button class="btn" id="btn-front">Pareto front</button>
          <button class="btn" id="btn-all">All</button>
          <button class="btn" id="btn-none">Clear</button>
          <button class="btn" id="btn-reset-names" hidden title="Back to the methods' own names">Reset names</button>
        </div>
        <div class="limits" id="limits" role="group" aria-label="Axis limits for this view">
          <span>Axis limits</span><span class="limits-head">min</span><span class="limits-head">max</span>
          <span>X</span>
          <input id="lim-xmin" type="number" step="any" min="0" aria-label="X-axis minimum">
          <input id="lim-xmax" type="number" step="any" min="0" aria-label="X-axis maximum">
          <span>Y</span>
          <input id="lim-ymin" type="number" step="any" aria-label="Y-axis minimum">
          <input id="lim-ymax" type="number" step="any" aria-label="Y-axis maximum">
          <div class="btnrow">
            <button class="btn" id="btn-auto-limits" title="Back to the automatic limits for this view">Auto limits</button>
          </div>
        </div>
        <span class="hint">Click methods or family buttons to highlight &middot; &#9998; on a method renames it &middot; hover points for details</span>
      </div>
      <div class="chips" id="chips"></div>
    </div>
    <div class="mainbox">
      <div class="exportbar" id="exportbar" hidden>
        <span class="hint">Export figure</span>
        <button class="btn" id="btn-svg" title="Download as SVG — vector, keeps text selectable">SVG</button>
        <button class="btn" id="btn-pdf" title="Download as a one-page PDF">PDF</button>
        <button class="btn" id="btn-png" title="Download as PNG at 3x scale">PNG</button>
      </div>
      <!-- Legend above the chart so readers decode the marks before the data. -->
      <div class="legendstrip" id="legendstrip"></div>
      <div class="chartbox" id="chartbox">
        <svg id="chart" role="img" aria-label="Pareto front explorer"></svg>
        <div class="tooltip"></div>
      </div>
    </div>
  </div>
  <details class="datatable">
    <summary>Data table</summary>
    <div class="tblwrap" id="tblwrap"></div>
  </details>

<script>
(function () {
  "use strict";
  const CONFIG = __CONFIG_JSON__;
  const POINTS = __POINTS_JSON__;

__BASE_JS__

  const TRAJECTORY = CONFIG.mode === "trajectory";

  // Task subsets the page holds (each point names its `subset`); one is shown at a time, the first at load.
  const SUBSETS = CONFIG.subsets || null;
  let subsetKey = SUBSETS ? SUBSETS[0].key : null;

  // The figure title, drawn at the top of the chart so that paper view and the exports carry it. Each
  // subset opens on its own default (the page's title and the subset's label); a title typed for a
  // subset stays with it, and an emptied box draws no title.
  const titleInput = document.getElementById("fig-title");
  const titleBySubset = new Map();
  function defaultTitle() {
    const subset = SUBSETS ? SUBSETS.find(s => s.key === subsetKey).label : null;
    return [CONFIG.title, subset].filter(Boolean).join(" · ");
  }
  function figTitle() { return titleBySubset.has(subsetKey) ? titleBySubset.get(subsetKey) : defaultTitle(); }
  function showTitle() { titleInput.value = figTitle(); }
  showTitle();

  // Column order: chips/controls left of the chart, or mirrored.
  document.getElementById("grid").classList.add(CONFIG.chipsSide === "right" ? "chips-right" : "chips-left");

  const svg = document.getElementById("chart");
  const box = document.getElementById("chartbox");
  const chipsBox = document.getElementById("chips");
  const controlsBox = document.querySelector(".controls");
  const tip = makeTooltip(box);
  // Matches the point labels drawn in `render` — see the side/overlap logic there.
  const textWidth = makeTextMeasurer(svg, { size: 13, weight: 700 });
  const axisTitleWidth = makeTextMeasurer(svg, { size: 14, weight: 650 });
  const figTitleWidth = makeTextMeasurer(svg, { size: 16, weight: 700 });

  // Marker glyph per variant at (cx, cy); trajectories use circles everywhere. `edge`
  // (a color, or null for none) outlines the glyph; the X glyph has no fill, so its edge
  // is a wider stroke beneath it.
  function drawMark(parent, cx, cy, variant, color, edge, size, opacity, dataM) {
    const common = { opacity: opacity, "data-m": dataM };
    const outline = edge ? { stroke: edge, "stroke-width": 1.2 } : {};
    let node;
    if (TRAJECTORY || variant === "Default" || !variant) {
      node = el("circle", { ...common, ...outline, cx, cy, r: size, fill: color }, parent);
    } else if (variant === "Tuned") {
      const s = size * 1.75;
      node = el("rect", { ...common, ...outline, x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1.5, fill: color }, parent);
    } else if (variant === "Tuned + Ens.") {
      const d = size * 0.95;
      const cross = {
        ...common,
        d: `M${cx - d},${cy - d} L${cx + d},${cy + d} M${cx - d},${cy + d} L${cx + d},${cy - d}`,
        fill: "none", "stroke-linecap": "round",
      };
      if (edge) el("path", { ...cross, stroke: edge, "stroke-width": size * 0.62 + 2.4 }, parent);
      node = el("path", { ...cross, stroke: color, "stroke-width": size * 0.62 }, parent);
    } else {
      // Any other variant (e.g. "End-to-end", holdout types): diamond.
      const s = size * 1.45;
      node = el("rect", {
        ...common, ...outline, x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1,
        fill: color, transform: `rotate(45 ${cx} ${cy})`,
      }, parent);
    }
    return node;
  }

  function drawImputedRing(parent, cx, cy, size, color, opacity, dataM) {
    el("circle", {
      cx, cy, r: size + 4.5, fill: "none", stroke: color, "stroke-width": 1.4,
      "stroke-dasharray": "3 2.5", opacity: opacity, "data-m": dataM,
    }, parent);
  }

  // Fat green arrow pointing into the optimal corner (mirrors the static
  // figures' "Optimal" arrow so both read the same way).
  function drawOptimalArrow(parent, lowerBetter, M, W, H) {
    const cornerY = lowerBetter ? H - M.b - 12 : M.t + 12;
    const tailY = lowerBetter ? H - M.b - 64 : M.t + 64;
    const cx = M.l + 12, tx = M.l + 64;
    const dx = cx - tx, dy = cornerY - tailY;
    const len = Math.hypot(dx, dy);
    const ux = dx / len, uy = dy / len;
    const headLen = 16;
    const bx = cx - ux * headLen, by = cornerY - uy * headLen; // head base center
    // line stops at the head base
    el("line", {
      x1: tx, y1: tailY, x2: bx, y2: by,
      stroke: "var(--optimal)", "stroke-width": 13, "stroke-linecap": "round", opacity: 0.92,
    }, parent);
    const px = -uy, py = ux; // perpendicular
    el("polygon", {
      points: `${cx},${cornerY} ${bx + px * 11},${by + py * 11} ${bx - px * 11},${by - py * 11}`,
      fill: "var(--optimal)", opacity: 0.92,
    }, parent);
    let angle = Math.atan2(dy, dx) * 180 / Math.PI;
    if (angle > 90 || angle < -90) angle += 180;
    const mx = (tx + bx) / 2, my = (tailY + by) / 2;
    const t = el("text", {
      x: mx, y: my, "text-anchor": "middle", "dominant-baseline": "middle",
      "font-size": 10.5, "font-weight": 700, fill: "#ffffff",
      transform: `rotate(${angle} ${mx} ${my})`,
    }, parent);
    t.textContent = "Optimal";
  }

  // ---------- data ----------
  // `POINTS` is everything the cell published; `SHOWN` is what the host lets the page keep
  // (see `onExcludeMessage`). Every computation below reads `SHOWN`, so an excluded method is
  // absent from the front, the axes and the chips alike.
  // With subsets, `SHOWN` also keeps only the current subset's points.
  const ALL_POINTS = POINTS.slice();
  let SHOWN = ALL_POINTS;
  let excludedMethods = new Set();
  const byMethod = new Map();
  function indexPoints() {
    byMethod.clear();
    for (const p of SHOWN) {
      if (!byMethod.has(p.method)) byMethod.set(p.method, []);
      byMethod.get(p.method).push(p); // insertion order = builder's point order
    }
  }
  function refilter() {
    SHOWN = ALL_POINTS.filter(p => (subsetKey === null || p.subset === subsetKey) && !excludedMethods.has(p.method));
    indexPoints();
  }
  refilter();

  // Display names: what the chart, chips, tooltips and table call a method. `p.method` stays the
  // method's key everywhere else (the front, the selection, the host's messages). The builder can
  // preset names (`CONFIG.methodNames`); a method renamed on its chip keeps the name on every subset
  // and view, and an empty name gives the method its own back.
  const names = new Map(Object.entries(CONFIG.methodNames || {}));
  function displayName(method) { return names.get(method) || method; }
  function escHtml(text) {
    return String(text).replace(/[&<>"']/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  const METRICS = CONFIG.metrics;
  let metricKey = METRICS[0].key;
  const metricByKey = {};
  for (const m of METRICS) metricByKey[m.key] = m;

  // Time axes offered by the x-axis selector (a single one hides it); the first is shown at
  // load. `xKey` is the current axis, `xAxis()` its spec.
  const X_AXES = CONFIG.xAxes;
  let xKey = X_AXES[0].key;
  const xAxisByKey = {};
  for (const a of X_AXES) xAxisByKey[a.key] = a;
  function xAxis() { return xAxisByKey[xKey]; }

  function mval(p, metric) { return p[metric.key]; }

  function computeFront(metric) {
    const xk = xKey;
    const pts = [...SHOWN].sort((a, b) =>
      a[xk] - b[xk] || (metric.lowerBetter ? mval(a, metric) - mval(b, metric) : mval(b, metric) - mval(a, metric)));
    const verts = [];
    const methods = new Set();
    const points = new Set();
    let best = null;
    for (const p of pts) {
      const v = mval(p, metric);
      if (best === null || (metric.lowerBetter ? v < best : v > best)) {
        if (best !== null) verts.push([p[xk], best]);
        verts.push([p[xk], v]);
        best = v;
        methods.add(p.method);
        points.add(p);
      }
    }
    return { verts, methods, points };
  }

  // The chart opens on the front, and keeps following it while the metric changes:
  // each metric has its own front (a method can lead on relative gain and be
  // mid-field on Elo), so a set carried over from the previous metric would leave
  // methods drawn on the front but greyed out. `followFront` drops as soon as the
  // reader picks methods themselves, so their selection is never overwritten.
  const state = { active: new Set(computeFront(metricByKey[metricKey]).methods), followFront: true };
  // Methods the host asks every chart to follow (`onFollowMessage`): drawn as active whatever the
  // chips select, and ringed. `hovered` is the method under the pointer in another chart of the
  // page (`onHoverMessage`), and `scale` the last render's axes, to ring it without a redraw.
  let followed = new Set();
  let hovered = null;
  let scale = null;
  let hoverMark = null;

  // ---------- axis limits ----------
  // Typed per view: each (subset, metric, time axis) keeps its own limits, so a range set for one
  // view stays with it while another is shown. A side left empty, or one the axis cannot use (x
  // must be above 0 on the log axis, and min below max), falls back to the automatic limit.
  const LIMIT_KEYS = ["xmin", "xmax", "ymin", "ymax"];
  const limitInputs = {};
  for (const k of LIMIT_KEYS) limitInputs[k] = document.getElementById("lim-" + k);
  const limitsByView = new Map();
  function viewKey() { return JSON.stringify([subsetKey, metricKey, xKey]); }
  function typedLimits() { return limitsByView.get(viewKey()) || {}; }
  // [lo, hi] from the typed limits where they make a range, else the automatic ones, and which
  // typed sides went unused.
  function axisRange(lo, hi, autoLo, autoHi, positive) {
    const usable = v => v != null && isFinite(v) && (!positive || v > 0);
    const a = usable(lo) ? lo : autoLo, b = usable(hi) ? hi : autoHi;
    if (a < b) return { lo: a, hi: b, badLo: lo != null && !usable(lo), badHi: hi != null && !usable(hi) };
    return { lo: autoLo, hi: autoHi, badLo: lo != null, badHi: hi != null };
  }
  function fmtLimit(v) {
    const a = Math.abs(v);
    return a >= 100 ? v.toFixed(0) : a >= 1 ? v.toFixed(2) : v.toPrecision(2);
  }
  // The boxes show the current view: its typed values and, as placeholders, the automatic limits
  // they would replace. A box holding text not yet entered (Enter or leaving it) keeps that text.
  function showLimits(auto, xr, yr) {
    const typed = typedLimits();
    const bad = { xmin: xr.badLo, xmax: xr.badHi, ymin: yr.badLo, ymax: yr.badHi };
    for (const k of LIMIT_KEYS) {
      const input = limitInputs[k];
      input.placeholder = fmtLimit(auto[k]);
      if (!input.dataset.editing) input.value = typed[k] == null ? "" : String(typed[k]);
      input.setAttribute("aria-invalid", String(bad[k]));
      input.title = bad[k] ? "Not used: x must be above 0, and min below max" : "";
    }
  }
  for (const k of LIMIT_KEYS) {
    limitInputs[k].addEventListener("input", ev => { ev.target.dataset.editing = "1"; });
    limitInputs[k].addEventListener("change", ev => {
      delete ev.target.dataset.editing;
      const text = ev.target.value.trim();
      const typed = { ...typedLimits(), [k]: text === "" ? null : Number(text) };
      if (LIMIT_KEYS.every(key => typed[key] == null)) limitsByView.delete(viewKey());
      else limitsByView.set(viewKey(), typed);
      render();
    });
  }
  titleInput.addEventListener("input", ev => {
    titleBySubset.set(subsetKey, ev.target.value);
    render();
  });
  document.getElementById("btn-auto-limits").addEventListener("click", () => {
    limitsByView.delete(viewKey());
    for (const k of LIMIT_KEYS) {
      delete limitInputs[k].dataset.editing;
      limitInputs[k].value = "";
    }
    render();
  });

  // ---------- chart ----------
  // A flat, fixed-height plot: two of these panels then fit on one screen.
  const CHART_H = 400;
  const M_PLOT = { l: 62, r: 18, t: 14, b: 52 };
  // The figure title's row above the plot, when there is a title.
  const TITLE_H = 30;
  let M = M_PLOT;

  function render() {
    // Nothing usable to measure yet: the frame is in a collapsed or hidden container, where
    // the box reports 0 and every string measures 0 too. Drawing now would pin the chart to
    // its minimum width and stack the labels on top of each other, and that is what the
    // reader would find when it finally appears. Wait instead — `redrawOnResize` calls back
    // the moment the box has a real width.
    if (!box.clientWidth) return;
    const metric = metricByKey[metricKey];
    svg.textContent = "";
    const W = Math.max(360, Math.round(box.clientWidth));
    const title = figTitle().trim();
    M = title ? { ...M_PLOT, t: M_PLOT.t + TITLE_H } : M_PLOT;
    const H = CHART_H + (title ? TITLE_H : 0);
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    // Keep the chip list from outgrowing the chart beside it.
    chipsBox.style.maxHeight = Math.max(170, H - controlsBox.offsetHeight + 20) + "px";

    // Automatic limits from the points shown, replaced by the view's typed ones where usable.
    const xsAll = SHOWN.map(p => p[xKey]);
    const vals = SHOWN.map(p => mval(p, metric));
    const auto = { xmin: Math.min(...xsAll) * 0.65, xmax: Math.max(...xsAll) * 1.6 };
    if (metric.fromZero) {
      auto.ymin = 0; auto.ymax = Math.max(...vals) * 1.07;
    } else {
      const pad = (Math.max(...vals) - Math.min(...vals)) * 0.07;
      auto.ymin = Math.min(...vals) - pad; auto.ymax = Math.max(...vals) + pad;
    }
    const typed = typedLimits();
    const xr = axisRange(typed.xmin, typed.xmax, auto.xmin, auto.xmax, true);
    const yr = axisRange(typed.ymin, typed.ymax, auto.ymin, auto.ymax, false);
    showLimits(auto, xr, yr);

    // x scale (log)
    const xmin = xr.lo, xmax = xr.hi;
    const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
    const X = v => M.l + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - M.l - M.r);
    const y0 = yr.lo, y1 = yr.hi;
    const Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * (H - M.t - M.b);
    const inView = p => p[xKey] >= xmin && p[xKey] <= xmax && mval(p, metric) >= y0 && mval(p, metric) <= y1;

    // grid + axes
    const grid = el("g", {}, svg);
    // Decades; on a range under two decades wide (e.g. a narrow typed range), 2x and 5x too.
    const steps = lx1 - lx0 < 2 ? [1, 2, 5] : [1];
    const xTicks = [];
    for (let e = Math.floor(lx0); e <= Math.ceil(lx1); e++) {
      for (const d of steps) {
        const v = d * Math.pow(10, e);
        if (v >= xmin && v <= xmax) xTicks.push({ v, e });
      }
    }
    for (const t of xTicks) {
      const gx = X(t.v);
      el("line", { x1: gx, y1: M.t, x2: gx, y2: H - M.b, stroke: "var(--line)", "stroke-width": 1 }, grid);
      const lbl = fmtNum(t.v, t.e >= 0 ? 0 : -t.e);
      el("text", { x: gx, y: H - M.b + 20, "text-anchor": "middle", "font-size": 12.5, fill: "var(--muted)" }, grid)
        .textContent = lbl;
    }
    for (const yv of ticks(y0, y1, 6)) {
      const gy = Y(yv);
      el("line", { x1: M.l, y1: gy, x2: W - M.r, y2: gy, stroke: "var(--line)", "stroke-width": 1 }, grid);
      el("text", { x: M.l - 8, y: gy + 4, "text-anchor": "end", "font-size": 12.5, fill: "var(--muted)" }, grid)
        .textContent = fmtNum(yv, Number.isInteger(yv) ? 0 : metric.decimals);
    }
    el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "none", stroke: "var(--line)" }, grid);
    // The axis title shortens where the chart is too narrow for it (a tile half the page wide):
    // first without the aggregation, then to the selector's own label.
    const full = xAxis().axisLabel;
    const titles = [full, full.replace(", median over datasets", ""), xAxis().label];
    const xTitle = titles.find(t => axisTitleWidth(t) <= W - M.l - M.r) || titles[titles.length - 1];
    el("text", {
      x: (M.l + W - M.r) / 2, y: H - 10, "text-anchor": "middle", "font-size": 14,
      "font-weight": 650, fill: "var(--ink)",
    }, grid).textContent = xTitle;
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": 14, "font-weight": 650, fill: "var(--ink)",
      transform: `translate(16 ${(M.t + H - M.b) / 2}) rotate(-90)`,
    }, grid).textContent = metric.axisLabel;

    if (title) {
      // Centred over the plot, and set smaller where the chart is too narrow for it.
      const room = W - M.l - M.r;
      const size = Math.max(11, Math.min(16, 16 * room / Math.max(1, figTitleWidth(title))));
      el("text", {
        x: (M.l + W - M.r) / 2, y: M_PLOT.t + 12, "text-anchor": "middle", "font-size": size,
        "font-weight": 700, fill: "var(--ink)",
      }, grid).textContent = title;
    }

    drawOptimalArrow(grid, metric.lowerBetter, M, W, H);

    // Everything drawn from the data sits in the plot's clip, so typed limits cut it at the frame
    // (and points outside them take no hover).
    const clip = el("clipPath", { id: "plotclip" }, el("defs", {}, svg));
    el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b }, clip);
    const plot = el("g", { "clip-path": "url(#plotclip)" }, svg);

    // pareto front (always shown)
    const front = computeFront(metric);
    const fv = front.verts;
    if (fv.length) {
      let d = `M${X(fv[0][0])},${metric.lowerBetter ? M.t : H - M.b}`;
      for (const [fx, fy] of fv) d += ` L${X(fx)},${Y(fy)}`;
      d += ` L${W - M.r},${Y(fv[fv.length - 1][1])}`;
      el("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6, "stroke-dasharray": "7 5", opacity: 0.85 }, plot);
    }

    const isOn = m => state.active.has(m) || followed.has(m);

    // connectors: variant links (scatter) / the trajectory itself
    const conn = el("g", {}, plot);
    for (const [method, pts] of byMethod) {
      if (pts.length < 2) continue;
      const on = isOn(method);
      if (!TRAJECTORY && !on) continue; // scatter: connectors only for active methods
      const dd = pts.map((p, i) => `${i ? "L" : "M"}${X(p[xKey])},${Y(mval(p, metric))}`).join(" ");
      el("path", {
        d: dd, fill: "none",
        stroke: on ? famColor(pts[0]) : "var(--pt-muted)",
        "stroke-width": on ? (TRAJECTORY ? 2 : 1.4) : 1,
        opacity: on ? 0.6 : 0.35,
        "data-m": method,
      }, conn);
    }

    // points: inactive first, active on top
    const ptsOff = el("g", {}, plot);
    const ptsOn = el("g", {}, plot);
    for (const [method, pts] of byMethod) {
      const on = isOn(method);
      for (const p of pts) {
        const color = on ? famColor(p) : "var(--pt-muted)";
        // An active marker's edge is a darker shade of its face on the light surface and
        // the card color on the dark one (the --fam-*-edge tokens). Muted markers stay
        // plain grey, easy to ignore.
        const edge = on ? famEdge(p) : null;
        // With `dimOffFront`, an active point that is not a vertex of the front is drawn
        // smaller and fainter, so the front members carry the eye.
        const dim = CONFIG.dimOffFront && on && !front.points.has(p);
        const size = (on ? (dim ? 5.5 : 7) : 5) * (TRAJECTORY ? 0.8 : 1);
        const op = on ? (dim ? 0.7 : 0.95) : 0.5;
        drawMark(on ? ptsOn : ptsOff, X(p[xKey]), Y(mval(p, metric)), p.variant, color, edge, size, op, p.method);
        // Imputation ring: every affected point in scatter mode; only the
        // trajectory's end point in trajectory mode (a ring on all ~8 line
        // points would read as beads, and the chip's ‡ already flags the line).
        if (p.imputed && (!TRAJECTORY || p === pts[pts.length - 1])) {
          drawImputedRing(on ? ptsOn : ptsOff, X(p[xKey]), Y(mval(p, metric)), size, color, op, p.method);
        }
      }
    }

    // Followed methods keep their family colour and gain a ring, on every chart of the page at
    // once; a trajectory rings its end point only, as the imputation ring does.
    const rings = el("g", { "pointer-events": "none" }, plot);
    for (const method of followed) {
      const pts = byMethod.get(method);
      if (!pts) continue;
      for (const p of TRAJECTORY ? pts.slice(-1) : pts) {
        el("circle", {
          cx: X(p[xKey]), cy: Y(mval(p, metric)), r: 10.5, fill: "none",
          stroke: "var(--accent)", "stroke-width": 2.2,
        }, rings);
      }
    }

    // Labels for active methods at their best point. Each one is measured, so it
    // can be put on whichever side of its point has room for the whole name and
    // held inside the plot — a system carries its configuration in its name
    // ("AutoGluon 1.6 (noncommercial, 4h)") and would otherwise run off the edge.
    const labels = [];
    for (const [method, pts] of byMethod) {
      if (!isOn(method)) continue;
      const best = pts.reduce((a, b) =>
        (metric.lowerBetter ? mval(a, metric) < mval(b, metric) : mval(a, metric) > mval(b, metric)) ? a : b);
      if (!inView(best)) continue;
      const px = X(best[xKey]);
      const w = textWidth(displayName(method));
      const toRight = px + 10 + w <= W - M.r;
      labels.push({
        method, family: best.family, shade: best.shade, w,
        anchor: toRight ? "start" : "end",
        // Anchored at its start the label runs right from `x`, at its end it runs
        // left to `x`; either way keep the far side off the plot frame.
        x: toRight ? px + 10 : Math.max(M.l + w + 2, px - 10),
        y: Y(mval(best, metric)) - 10,
      });
    }
    // Greedy de-overlap: push a label down a line whenever it would sit on top of
    // one already placed. Measured spans, so a long name displaces its neighbours
    // for as far as it actually reaches.
    const spanOf = l => (l.anchor === "start" ? [l.x, l.x + l.w] : [l.x - l.w, l.x]);
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      for (let j = 0; j < i; j++) {
        const [aL, aR] = spanOf(labels[i]), [bL, bR] = spanOf(labels[j]);
        if (aL < bR + 6 && bL < aR + 6 && Math.abs(labels[i].y - labels[j].y) < 15) {
          labels[i].y = labels[j].y + 15;
        }
      }
    }
    const lg = el("g", {}, svg);
    for (const l of labels) {
      const t = el("text", {
        x: l.x, y: Math.max(l.y, M.t + 12), "font-size": 13, "font-weight": 700,
        fill: famColor(l), "paint-order": "stroke", stroke: "var(--card)", "stroke-width": 3.5,
        "text-anchor": l.anchor,
      }, lg);
      t.textContent = displayName(l.method);
    }

    // invisible hit targets on top (bigger than marks)
    const hits = el("g", {}, plot);
    for (const p of SHOWN) {
      const h = el("circle", { cx: X(p[xKey]), cy: Y(mval(p, metric)), r: 12, fill: "transparent", cursor: "pointer" }, hits);
      h.addEventListener("mouseenter", ev => { showTip(p, ev); postHover(p.method); });
      h.addEventListener("mousemove", ev => tip.move(ev));
      h.addEventListener("mouseleave", () => { hideTip(p.method); postHover(null); });
      h.addEventListener("click", () => pickMethod(p.method, () => toggle(p.method)));
    }
    scale = { X, Y, metric, plot };
    hoverMark = null;
    markHovered();
  }

  // Ring the method under the pointer in another chart of the page, dashed so it does not read
  // as followed. Drawn on top, but without hit area, so it never takes this chart's own hover.
  function markHovered() {
    if (hoverMark) hoverMark.remove();
    hoverMark = null;
    const pts = hovered && scale ? byMethod.get(hovered) : null;
    if (!pts) return;
    hoverMark = el("g", { "pointer-events": "none" }, scale.plot);
    for (const p of pts) {
      el("circle", {
        cx: scale.X(p[xKey]), cy: scale.Y(mval(p, scale.metric)), r: 9.5, fill: "none",
        stroke: "var(--ink)", "stroke-width": 1.8, "stroke-dasharray": "3 2.5",
      }, hoverMark);
    }
  }
  onHoverMessage(method => {
    hovered = method;
    markHovered();
  });
  onFollowMessage(names => {
    followed = names;
    render();
  });

  // Temporary hover emphasis without a re-render (a re-render would replace
  // the hit node under the cursor mid-hover).
  function emphasize(method, on) {
    svg.querySelectorAll(`[data-m="${CSS.escape(method)}"]`).forEach(n => {
      if (on) {
        if (!n.dataset.save) n.dataset.save = n.getAttribute("opacity") || "1";
        n.setAttribute("opacity", "0.95");
      } else if (n.dataset.save) {
        n.setAttribute("opacity", n.dataset.save);
        delete n.dataset.save;
      }
    });
  }

  function showTip(p, ev) {
    emphasize(p.method, true);
    const sub = TRAJECTORY ? (p.n_configs != null ? `${p.n_configs} configs` : "") : (p.variant || "");
    let html = `<div class="t-name">${escHtml(displayName(p.method))}` + (sub ? ` <span class="t-var">(${sub})</span>` : "") + "</div>" +
      (names.has(p.method) ? `<div class="t-var">${escHtml(p.method)}</div>` : "") +
      `<div>${famLabel(p)}</div>`;
    for (const m of METRICS) {
      html += `<div>${m.label}: <b>${fmtMetric(m, mval(p, m))}</b></div>`;
    }
    html += `<div>${xAxis().short}: <b>${fmtTime(p[xKey])}</b></div>`;
    if (p.imputed) html += `<div class="t-imp">Imputed on ${fmtNum(p.imputed_pct, 0)}% of datasets</div>`;
    tip.show(html + PICK_HINT, ev);
  }
  function hideTip(method) {
    emphasize(method, false);
    tip.hide();
  }

  // ---------- chips ----------
  const chipByMethod = new Map();
  const famChips = new Map();
  function familyMethods(fam) {
    return [...byMethod.keys()].filter(m => byMethod.get(m)[0].family === fam);
  }
  // Chips are listed by leaderboard rank — best Elo first when Elo is
  // configured, otherwise best value of the primary metric.
  const RANK_METRIC = metricByKey["elo"] || metricByKey[METRICS[0].key];
  function bestVal(method) {
    const vals = byMethod.get(method).map(p => mval(p, RANK_METRIC));
    return RANK_METRIC.lowerBetter ? Math.min(...vals) : Math.max(...vals);
  }
  function rankSorted(methods) {
    return [...methods].sort((a, b) =>
      RANK_METRIC.lowerBetter ? bestVal(a) - bestVal(b) : bestVal(b) - bestVal(a));
  }
  function buildChips() {
    for (const fam of FAM_ORDER) {
      const methods = rankSorted(familyMethods(fam));
      if (!methods.length) continue;
      const row = document.createElement("div");
      row.className = "chiprow";
      const famBtn = document.createElement("button");
      famBtn.className = "famchip";
      famBtn.style.setProperty("--fam", FAM_VAR[fam]);
      famBtn.innerHTML = famChipLabel(fam, methods.length);
      famBtn.title = `Toggle all ${methods.length} ${fam} methods`;
      famBtn.addEventListener("click", () => toggleFamily(fam));
      row.appendChild(famBtn);
      famChips.set(fam, famBtn);
      const set = document.createElement("div");
      set.className = "chipset";
      for (const m of methods) {
        const wrap = document.createElement("span");
        wrap.className = "chipwrap";
        const b = document.createElement("button");
        b.className = "chip";
        b.style.setProperty("--fam", famColor(byMethod.get(m)[0]));
        const label = document.createElement("span");
        label.className = "chip-label";
        b.appendChild(Object.assign(document.createElement("span"), { className: "dot" }));
        b.appendChild(label);
        if (byMethod.get(m).some(p => p.imputed)) {
          const mark = document.createElement("span");
          mark.className = "imp-mark";
          mark.textContent = "‡";
          b.appendChild(mark);
        }
        b.addEventListener("click", () => toggle(m));
        const edit = document.createElement("button");
        edit.className = "chip-edit";
        edit.textContent = "✎";
        edit.addEventListener("click", ev => { ev.stopPropagation(); startRename(m, wrap, b, edit); });
        wrap.appendChild(b);
        wrap.appendChild(edit);
        set.appendChild(wrap);
        chipByMethod.set(m, b);
        nameChip(m);
      }
      row.appendChild(set);
      chipsBox.appendChild(row);
    }
  }
  // A chip's label and hover text from the method's display name.
  function nameChip(m) {
    const b = chipByMethod.get(m);
    const imputed = byMethod.has(m) && byMethod.get(m).some(p => p.imputed);
    b.querySelector(".chip-label").textContent = displayName(m);
    b.title = displayName(m) + (names.has(m) ? ` (${m})` : "") + (imputed ? " — partially imputed" : "");
    b.parentElement.querySelector(".chip-edit").title = `Rename ${displayName(m)}`;
    b.parentElement.querySelector(".chip-edit").setAttribute("aria-label", `Rename ${displayName(m)}`);
  }
  // The rename box opens in the chip's place: Enter or leaving the box applies the name, Escape
  // keeps the old one. Both keys stop at the box (Escape elsewhere opens paper view, and an Enter
  // reaching the chip would toggle it); closed from the keyboard, the box hands the focus back to its
  // chip on the next frame.
  function startRename(m, wrap, chip, edit) {
    const input = document.createElement("input");
    input.className = "chip-rename";
    input.type = "text";
    input.spellcheck = false;
    input.value = displayName(m);
    input.placeholder = m;
    input.size = Math.max(9, input.value.length + 1);
    input.setAttribute("aria-label", `New name for ${m}`);
    chip.hidden = true;
    edit.hidden = true;
    wrap.insertBefore(input, chip);
    input.focus();
    input.select();
    let open = true;
    function close(apply, refocus) {
      if (!open) return;
      open = false;
      input.remove();
      chip.hidden = false;
      edit.hidden = false;
      if (apply) setName(m, input.value);
      if (refocus) requestAnimationFrame(() => chip.focus());
    }
    input.addEventListener("keydown", ev => {
      if (ev.key !== "Enter" && ev.key !== "Escape") return;
      ev.preventDefault();
      ev.stopPropagation();
      close(ev.key === "Enter", true);
    });
    input.addEventListener("blur", () => close(true, false));
  }
  const resetNamesBtn = document.getElementById("btn-reset-names");
  function setName(m, text) {
    const name = text.trim();
    if (!name || name === m) names.delete(m); else names.set(m, name);
    refreshNames();
  }
  function refreshNames() {
    for (const m of chipByMethod.keys()) nameChip(m);
    resetNamesBtn.hidden = !names.size;
    buildTable();
    render();
  }
  resetNamesBtn.addEventListener("click", () => {
    names.clear();
    refreshNames();
  });
  resetNamesBtn.hidden = !names.size;

  function syncChips() {
    for (const [m, b] of chipByMethod) {
      b.setAttribute("aria-pressed", String(state.active.has(m)));
      b.parentElement.hidden = !byMethod.has(m);
    }
    for (const [fam, b] of famChips) {
      const methods = familyMethods(fam);
      b.hidden = !methods.length;
      b.setAttribute("aria-pressed", String(methods.every(m => state.active.has(m))));
      b.innerHTML = famChipLabel(fam, methods.length);
    }
  }
  // The host's exclusion list (see `onExcludeMessage` in the shared script). Re-indexes the
  // points and drops the excluded methods from the active set; a chart that was following
  // the front recomputes it over what is left.
  onExcludeMessage(excluded => {
    excludedMethods = excluded;
    refilter();
    for (const m of [...state.active]) if (!byMethod.has(m)) state.active.delete(m);
    if (state.followFront) state.active = new Set(computeFront(metricByKey[metricKey]).methods);
    syncChips();
    render();
  });
  function toggle(m) {
    if (state.active.has(m)) state.active.delete(m); else state.active.add(m);
    state.followFront = false;
    syncChips();
    render();
  }
  function toggleFamily(fam) {
    const methods = familyMethods(fam);
    const allOn = methods.every(m => state.active.has(m));
    for (const m of methods) {
      if (allOn) state.active.delete(m); else state.active.add(m);
    }
    state.followFront = false;
    syncChips();
    render();
  }
  function setActive(methods, followFront = false) {
    state.active = new Set(methods);
    state.followFront = followFront;
    syncChips();
    render();
  }
  function showFront() {
    setActive(computeFront(metricByKey[metricKey]).methods, true);
  }
  // Switching an axis moves the front, so the highlighted set moves with it.
  function setMetric(key) {
    metricKey = key;
    if (state.followFront) showFront(); else render();
  }
  function setXAxis(key) {
    xKey = key;
    buildTable();
    if (state.followFront) showFront(); else render();
  }
  // Another subset brings its own points, so the chips, legend and table are rebuilt; the highlighted
  // methods stay where the subset has them.
  function setSubset(key) {
    subsetKey = key;
    refilter();
    chipsBox.innerHTML = "";
    chipByMethod.clear();
    famChips.clear();
    buildChips();
    for (const m of [...state.active]) if (!byMethod.has(m)) state.active.delete(m);
    if (state.followFront) state.active = new Set(computeFront(metricByKey[metricKey]).methods);
    showTitle();
    buildLegend();
    buildTable();
    syncChips();
    render();
  }
  document.getElementById("btn-front").addEventListener("click", showFront);
  document.getElementById("btn-all").addEventListener("click", () => setActive([...byMethod.keys()]));
  document.getElementById("btn-none").addEventListener("click", () => setActive([]));

  // metric selector (hidden when only one metric is configured)
  const metricPick = document.getElementById("metricpick");
  const metricSelect = document.getElementById("metric-select");
  if (METRICS.length > 1) {
    metricPick.hidden = false;
    for (const m of METRICS) {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.label;
      metricSelect.appendChild(opt);
    }
    metricSelect.addEventListener("change", ev => setMetric(ev.target.value));
  }

  // subset selector (hidden when the page holds no subsets)
  const subsetPick = document.getElementById("subsetpick");
  const subsetSelect = document.getElementById("subset-select");
  if (SUBSETS) {
    subsetPick.hidden = false;
    for (const s of SUBSETS) {
      const opt = document.createElement("option");
      opt.value = s.key;
      opt.textContent = s.label;
      subsetSelect.appendChild(opt);
    }
    subsetSelect.addEventListener("change", ev => setSubset(ev.target.value));
  }

  // x-axis selector (hidden when only one time axis is configured); axes that name a `group`
  // (e.g. "Median, per 1K samples") are listed under it
  const xAxisPick = document.getElementById("xaxispick");
  const xAxisSelect = document.getElementById("xaxis-select");
  if (X_AXES.length > 1) {
    xAxisPick.hidden = false;
    const groups = new Map();
    for (const a of X_AXES) {
      const opt = document.createElement("option");
      opt.value = a.key;
      opt.textContent = a.label;
      if (a.group) {
        if (!groups.has(a.group)) {
          const og = document.createElement("optgroup");
          og.label = a.group;
          groups.set(a.group, og);
          xAxisSelect.appendChild(og);
        }
        groups.get(a.group).appendChild(opt);
      } else {
        xAxisSelect.appendChild(opt);
      }
    }
    xAxisSelect.addEventListener("change", ev => setXAxis(ev.target.value));
  }

  // Embedded, the host can pick the metric for us: the leaderboard's "I care about" control
  // decides whether the page leads with Elo or Improvability, and every panel follows without
  // regenerating an artifact per metric. Ignored when the metric is not one this chart offers.
  // The host re-sends its choice every time the frame reports a height, and a metric picked in
  // here (or a paper-view toggle) changes the height. So a repeat of the host's last message is
  // ignored outright, rather than compared with the metric on screen: otherwise the reader's
  // own pick would be undone by the echo of a choice the host had already made.
  // The host can pick the time axis the same way (``tabarena-explorer-xaxis``).
  let lastHostMetric = null;
  let lastHostXAxis = null;
  window.addEventListener("message", ev => {
    const d = ev.data;
    if (d && d.type === "tabarena-explorer-xaxis") {
      if (d.xaxis === lastHostXAxis) return;
      lastHostXAxis = d.xaxis;
      if (!X_AXES.some(a => a.key === d.xaxis) || d.xaxis === xKey) return;
      if (xAxisSelect) xAxisSelect.value = d.xaxis;
      setXAxis(d.xaxis);
      return;
    }
    if (!d || d.type !== "tabarena-explorer-metric") return;
    if (d.metric === lastHostMetric) return;
    lastHostMetric = d.metric;
    if (!METRICS.some(m => m.key === d.metric) || d.metric === metricKey) return;
    if (metricSelect) metricSelect.value = d.metric;
    setMetric(d.metric);
  });

  // ---------- legend strip ----------
  function buildLegend() {
    const box2 = document.getElementById("legendstrip");
    let html = "";
    if (!TRAJECTORY) {
      html +=
        '<span class="item"><svg width="14" height="14" viewBox="0 0 14 14"><circle cx="7" cy="7" r="5" fill="var(--muted)"/></svg> Default</span>' +
        '<span class="item"><svg width="14" height="14" viewBox="0 0 14 14"><rect x="2" y="2" width="10" height="10" rx="1.5" fill="var(--muted)"/></svg> Tuned</span>' +
        '<span class="item"><svg width="14" height="14" viewBox="0 0 14 14"><path d="M3,3 L11,11 M3,11 L11,3" stroke="var(--muted)" stroke-width="2.6" stroke-linecap="round"/></svg> Tuned + Ensembled</span>';
      // Systems draw as diamonds; the legend names them as the leaderboard table does.
      if (SHOWN.some(p => p.variant === "End-to-end")) {
        html += '<span class="item" title="A system picked, tuned and combined its own models inside one budget; the tuning regimes do not apply to it"><svg width="14" height="14" viewBox="0 0 14 14"><rect x="3" y="3" width="8" height="8" rx="1" fill="var(--muted)" transform="rotate(45 7 7)"/></svg> End-to-end</span>';
      }
    } else {
      html += '<span class="item"><svg width="26" height="8" viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="var(--muted)" stroke-width="2"/><circle cx="6" cy="4" r="2.6" fill="var(--muted)"/><circle cx="16" cy="4" r="2.6" fill="var(--muted)"/></svg> Tuning trajectory (more configs &rarr; more time)</span>';
    }
    html += '<span class="item"><svg width="26" height="8" viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="var(--ink)" stroke-width="1.6" stroke-dasharray="6 4"/></svg> Pareto front (always shown)</span>';
    if (SHOWN.some(p => p.imputed)) {
      html += '<span class="item"><svg width="18" height="18" viewBox="0 0 18 18"><circle cx="9" cy="9" r="4" fill="var(--muted)"/><circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--muted)" stroke-width="1.3" stroke-dasharray="3 2.5"/></svg> &Dagger; partially imputed</span>';
    }
    // Model family, named: highlighted points are colored by family, and in paper
    // view the chip list that would otherwise decode them is hidden.
    const swatches = legendSwatches(SHOWN);
    if (swatches.length > 1) {
      html += '<span class="legendbreak"></span><span class="item">Family:</span>';
      for (const sw of swatches) {
        html += `<span class="item"><svg width="12" height="12" viewBox="0 0 12 12">` +
          `<circle cx="6" cy="6" r="5" fill="${sw.color}"/></svg> ` +
          `<span style="color:${sw.ink}">${sw.label}</span></span>`;
      }
    }
    box2.innerHTML = html;
  }

  // ---------- paper view ----------
  setUpPaperView(render);
  setUpExport(() => [{ svg: svg, dx: 0 }], () => slugify(figTitle().trim() || document.title));

  // ---------- data table ----------
  function buildTable() {
    const m0 = metricByKey[METRICS[0].key];
    const rows = [...SHOWN].sort((a, b) =>
      m0.lowerBetter ? mval(a, m0) - mval(b, m0) : mval(b, m0) - mval(a, m0));
    let html = "<table><thead><tr><th>Method</th>";
    html += TRAJECTORY ? "<th>Configs</th>" : "<th>Variant</th>";
    html += "<th>Family</th>";
    for (const m of METRICS) html += `<th>${m.label}</th>`;
    html += `<th>${xAxis().short}</th><th>Imputed</th></tr></thead><tbody>`;
    for (const p of rows) {
      html += `<tr><td>${escHtml(displayName(p.method))}</td><td>${TRAJECTORY ? (p.n_configs != null ? p.n_configs : "—") : p.variant}</td><td>${p.family}</td>`;
      for (const m of METRICS) html += `<td>${fmtMetric(m, mval(p, m))}</td>`;
      html += `<td>${fmtNum(p[xKey], 3)}</td>`;
      html += `<td>${p.imputed ? fmtNum(p.imputed_pct, 0) + "%" : "—"}</td></tr>`;
    }
    html += "</tbody></table>";
    document.getElementById("tblwrap").innerHTML = html;
  }

  const _renderInner = render;
  render = function () {
    _renderInner();
    postHeight();
  };
  document.querySelector("details.datatable").addEventListener("toggle", postHeight);
  // Watches the chart's own box, not just the viewport: a frame laid out after it loaded (the
  // reader scrolled past before it painted) changes this element's width without ever changing
  // the viewport, and that is the case that leaves a chart stuck at its minimum width.
  redrawOnResize(box, render);
  window.addEventListener("load", postHeight);

  buildChips();
  buildLegend();
  buildTable();
  syncChips();
  render();
})();
</script>
</body>
</html>
"""
