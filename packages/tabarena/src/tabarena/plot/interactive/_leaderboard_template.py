"""HTML/CSS/JS template for the interactive leaderboard-overview explorer.

The interactive twin of the static ``tuning-impact-elo`` bar figure: one column
group per method, nested bars for the tuning variants (default / tuned /
tuned + ensembled), bootstrap CI whiskers, and reference pipelines as horizontal
threshold lines. On top of the static figure it offers a metric selector, sorting,
per-method and per-family removal, hover values and a data table — and it scrolls
horizontally instead of shrinking 40+ methods into one page-wide strip.

Kept as a Python string constant (rather than a package-data file) so it ships
with the package without any build-system data-file configuration.
:func:`tabarena.plot.interactive.leaderboard_explorer.build_leaderboard_explorer_html`
substitutes the placeholders (see
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

LEADERBOARD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  /* Chart: a fixed y-axis pane beside a horizontally scrolling plot pane, so
     the axis stays readable however far right the reader scrolls. */
  .lb-chartwrap { position: relative; display: flex; align-items: flex-start; }
  .lb-axis { flex: none; }
  .lb-scroll {
    flex: 1 1 auto; min-width: 0; overflow-x: auto; overflow-y: hidden;
    scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent;
  }
  .lb-scroll::-webkit-scrollbar { height: 10px; }
  .lb-scroll::-webkit-scrollbar-thumb {
    background: var(--pt-muted); border-radius: 8px;
    border: 3px solid transparent; background-clip: content-box;
  }
  .rangepick input[type=range] { width: 96px; accent-color: var(--accent); cursor: pointer; }
  .rangeval { font-variant-numeric: tabular-nums; font-weight: 500; min-width: 3.4em; }
  .lb-empty { padding: 40px 0; text-align: center; color: var(--muted); font-size: 13px; }
  /* Puts the family key on its own line of the legend strip. */
  .legendstrip .legendbreak { flex-basis: 100%; height: 0; }
  /* Family blocks pack side by side rather than stacking: below a full-width
     chart, one row per family leaves the small ones (Other, Baseline) each
     wasting a line. The wide families still claim a row of their own. */
  .chips { margin-top: 10px; flex-direction: row; flex-wrap: wrap; gap: 12px 28px; align-items: flex-start; }
  .chips-head { flex: 1 1 100%; font-size: 12.5px; color: var(--muted); font-weight: 600; }
  .chiprow { flex: 0 1 auto; min-width: 0; }
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, chart and legend only — for slides and papers">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>
  <!-- One control row above everything it scopes (axis, sorting, variants, selection). -->
  <div class="controls">
    <label class="metricpick">Y-axis
      <select id="metric-select"></select>
    </label>
    <label class="metricpick">Sort
      <select id="sort-select"></select>
    </label>
    <label class="metricpick rangepick" title="Raise the axis floor to spread out the top of the field">Zoom
      <input type="range" id="ymin-range" aria-label="Y-axis minimum">
      <span class="rangeval" id="ymin-val"></span>
    </label>
    <div class="btnrow" id="variant-btns"></div>
    <div class="btnrow">
      <button class="btn" id="btn-all">All methods</button>
      <button class="btn" id="btn-top">Top 15</button>
      <button class="btn" id="btn-none">Clear</button>
    </div>
    <span class="hint">Click a column or chip to remove a method &middot; hover for exact values</span>
  </div>
  <div class="exportbar" id="exportbar" hidden>
    <span class="hint">Export figure</span>
    <button class="btn" id="btn-svg" title="Download as SVG — vector, keeps text selectable">SVG</button>
    <button class="btn" id="btn-pdf" title="Download as a one-page PDF">PDF</button>
    <button class="btn" id="btn-png" title="Download as PNG at 3x scale">PNG</button>
  </div>
  <div class="legendstrip" id="legendstrip"></div>
  <div class="lb-chartwrap" id="chartwrap">
    <svg id="axis" class="lb-axis" role="presentation"></svg>
    <div class="lb-scroll" id="scroller">
      <svg id="chart" role="img" aria-label="Leaderboard overview"></svg>
    </div>
    <div class="tooltip"></div>
  </div>
  <div class="chips" id="chips"></div>
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

  // ---------- geometry ----------
  const AXIS_W = 72;      // width of the sticky y-axis pane
  const PLOT_H = 320;     // height of the plot area itself
  const TOP = 14;         // headroom above the tallest bar
  const LABEL_TOP = 18;   // axis line -> first label row
  const LABEL_ROW = 19;   // vertical offset of the staggered second row
  const LABEL_SIZE = 14;  // method names; the slot below scales with it
  const TICK_SIZE = 12.5;
  // Method names are set horizontally (rotated ones are markedly harder to
  // read), staggered over two rows exactly like the static figure — so a slot
  // has to be wide enough for half a name. Past ~24 methods the chart scrolls
  // rather than squeezing every column into the viewport.
  // No upper bound on the slot: with few methods selected a capped slot left
  // the columns huddled on the left with dead space beside them, so the slots
  // simply share out whatever width there is.
  const MIN_SLOT = 66;
  const BAR_FRAC = 0.88;  // share of the slot the widest bar takes
  const MAX_BAR = 70;     // ...but never wider than this, however few columns

  // Variant -> (color token, width relative to the widest bar). The bars are
  // concentric, so the nesting itself encodes the tuning progression.
  const VARIANT_STYLE = {
    "Tuned + Ens.": { color: "var(--var-tunedens)", rel: 1 },
    "Tuned": { color: "var(--var-tuned)", rel: 0.8 },
    "Default": { color: "var(--var-default)", rel: 0.6 },
  };
  const VARIANT_ORDER = ["Default", "Tuned", "Tuned + Ens."];
  // Dash patterns cycle so several reference lines stay distinguishable.
  const REF_DASHES = ["8 5", "2 4", "12 4 3 4"];

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;

  const axisSvg = document.getElementById("axis");
  const svg = document.getElementById("chart");
  const scroller = document.getElementById("scroller");
  const wrap = document.getElementById("chartwrap");
  const yminRange = document.getElementById("ymin-range");
  const yminVal = document.getElementById("ymin-val");
  const tip = makeTooltip(wrap);

  // ---------- data ----------
  const METRICS = CONFIG.metrics;
  const metricByKey = {};
  for (const m of METRICS) metricByKey[m.key] = m;

  // One entry per method (its variants grouped); reference pipelines are kept
  // apart — they are drawn as threshold lines, not as columns.
  const byMethod = new Map();
  const refs = [];
  for (const p of POINTS) {
    if (p.reference) { refs.push(p); continue; }
    let entry = byMethod.get(p.method);
    if (!entry) {
      entry = { method: p.method, family: p.family, url: p.url, points: [] };
      byMethod.set(p.method, entry);
    }
    entry.points.push(p);
  }
  for (const e of byMethod.values()) {
    e.points.sort((a, b) => VARIANT_ORDER.indexOf(a.variant) - VARIANT_ORDER.indexOf(b.variant));
    e.imputed = e.points.some(p => p.imputed);
    e.imputed_pct = Math.max(...e.points.map(p => p.imputed_pct || 0));
  }

  const state = {
    metric: METRICS[0].key,
    sort: "best",
    yMin: null,   // null = the automatic axis floor; a number = zoomed in

    methods: new Set(byMethod.keys()),
    refs: new Set(refs.map(r => r.method)),
    variants: new Set(VARIANT_ORDER),
  };

  function metric() { return metricByKey[state.metric]; }

  // A method's best value under `m`, ignoring variants the reader switched off.
  function bestOf(entry, m) {
    const vals = entry.points
      .filter(p => state.variants.has(p.variant) && p[m.key] != null)
      .map(p => p[m.key]);
    if (!vals.length) return null;
    return m.lowerBetter ? Math.min(...vals) : Math.max(...vals);
  }

  function sortedMethods(entries, m) {
    const arr = [...entries];
    const cmp = {
      best: (a, b) => rankVal(a, m) - rankVal(b, m),
      worst: (a, b) => rankVal(b, m) - rankVal(a, m),
      name: (a, b) => a.method.localeCompare(b.method),
      family: (a, b) =>
        FAM_ORDER.indexOf(a.family) - FAM_ORDER.indexOf(b.family) || rankVal(a, m) - rankVal(b, m),
    }[state.sort];
    return arr.sort(cmp);
  }
  // Sort key that puts "better" first for either metric direction, with
  // value-less methods last.
  function rankVal(entry, m) {
    const v = bestOf(entry, m);
    if (v == null) return Infinity;
    return m.lowerBetter ? v : -v;
  }

  function visibleEntries() {
    const m = metric();
    return sortedMethods([...byMethod.values()].filter(e => state.methods.has(e.method) && bestOf(e, m) != null), m);
  }
  function visibleRefs() {
    return refs.filter(r => state.refs.has(r.method) && r[state.metric] != null);
  }

  // Rendered width of each label, measured in the live document (font metrics
  // are not knowable up front): it decides whether the names fit on one row or
  // need the two-row stagger, and which ones have to be shortened.
  function measureLabels(names) {
    const probe = el("g", { visibility: "hidden" }, svg);
    const widths = names.map(name => {
      const t = el("text", { "font-size": LABEL_SIZE }, probe);
      t.textContent = name;
      return t.getComputedTextLength();
    });
    probe.remove();
    return widths;
  }

  // Trim a name to `budget` px, ending in an ellipsis. The full name stays one
  // hover (and one data-table row) away.
  function fitLabel(node, name, width, budget) {
    if (width <= budget) return;
    let text = name;
    while (text.length > 1 && node.getComputedTextLength() > budget) {
      text = text.slice(0, -1);
      node.textContent = text + "…";
    }
  }

  // ---------- chart ----------
  function render() {
    const m = metric();
    const entries = visibleEntries();
    const shownRefs = visibleRefs();
    svg.textContent = "";
    axisSvg.textContent = "";
    axisSvg.setAttribute("width", AXIS_W);

    const avail = Math.max(240, scroller.clientWidth - 2);
    if (!entries.length) {
      axisSvg.setAttribute("height", 120);
      svg.setAttribute("width", avail);
      svg.setAttribute("height", 120);
      const t = el("text", { x: avail / 2, y: 60, "text-anchor": "middle", "font-size": 13, fill: "var(--muted)" }, svg);
      t.textContent = "No methods selected — use “All methods” to bring them back.";
      buildLegend(m, shownRefs);
      postHeight();
      return;
    }

    const slot = Math.max(MIN_SLOT, avail / entries.length);
    const barUnit = Math.min(MAX_BAR, slot * BAR_FRAC);
    const plotW = Math.max(avail, slot * entries.length);
    // Names go on one row when they fit side by side, otherwise on two
    // staggered rows (each label then has two slots of room).
    const labels = entries.map(e => e.method + (e.imputed ? " ‡" : ""));
    const labelWidths = measureLabels(labels);
    const widest = Math.max(...labelWidths);
    const labelRows = widest <= slot - 6 ? 1 : 2;
    const H = TOP + PLOT_H + LABEL_TOP + (labelRows - 1) * LABEL_ROW + 10;
    svg.setAttribute("width", plotW);
    svg.setAttribute("height", H);
    axisSvg.setAttribute("height", H);

    // -- y domain: from zero where the metric has one, else a floor below the
    //    shortest bar (Elo has no meaningful zero, and starting at zero would
    //    squash every difference into the top fifth of the chart). The floor
    //    ignores the CI whiskers so one wide interval cannot deflate the scale.
    const barVals = [];
    const allVals = [];
    for (const e of entries) {
      for (const p of e.points) {
        if (!state.variants.has(p.variant) || p[m.key] == null) continue;
        barVals.push(p[m.key]);
        allVals.push(p[m.key]);
        if (m.ci && p[m.ci.hi] != null) allVals.push(p[m.ci.hi]);
      }
    }
    for (const r of shownRefs) { barVals.push(r[state.metric]); allVals.push(r[state.metric]); }
    const barMin = Math.min(...barVals), barMax = Math.max(...allVals);
    const span = barMax - barMin || Math.abs(barMax) || 1;
    let autoY0, y1;
    if (m.fromZero) {
      autoY0 = 0;
      y1 = barMax * 1.06 || 1;
    } else {
      // Snap the floor down to a tick multiple so the axis reads in round
      // numbers, and keep it just below the shortest bar: every bar stays
      // visible (the static figure clips the ones below its fixed floor).
      const step = niceStep(span / 6);
      autoY0 = Math.floor((barMin - span * 0.05) / step) * step;
      y1 = barMax + span * 0.04;
    }
    // The zoom slider raises the floor from there toward the top of the field,
    // magnifying the differences between the leaders. Its bounds follow the
    // metric's own scale, so they are refreshed on every render; the bars and
    // whiskers that drop below the new floor are clipped away (see plot-clip).
    const zoomMax = autoY0 + (y1 - autoY0) * 0.9;
    yminRange.min = autoY0;
    yminRange.max = zoomMax;
    yminRange.step = (zoomMax - autoY0) / 200 || 1;
    if (state.yMin != null) state.yMin = Math.min(state.yMin, zoomMax);
    if (state.yMin != null && state.yMin <= autoY0) state.yMin = null;
    const y0 = state.yMin == null ? autoY0 : state.yMin;
    yminRange.value = y0;
    yminVal.textContent = state.yMin == null ? "auto" : fmtNum(y0, m.decimals);
    const Y = v => TOP + (1 - (v - y0) / (y1 - y0)) * PLOT_H;
    const baseY = Y(y0);

    // -- grid (solid hairlines) in the plot pane, tick labels in the axis pane
    const tickVals = ticks(y0, y1, 6);
    const grid = el("g", {}, svg);
    const tickLabels = [];
    for (const tv of tickVals) {
      const gy = Y(tv);
      el("line", { x1: 0, y1: gy, x2: plotW, y2: gy, stroke: "var(--line)", "stroke-width": 1 }, grid);
      const t = el("text", {
        x: AXIS_W - 10, y: gy + 4, "text-anchor": "end", "font-size": TICK_SIZE, fill: "var(--muted)",
      }, axisSvg);
      t.textContent = fmtNum(tv, m.decimals);
      tickLabels.push({ y: gy, node: t });
    }
    el("line", { x1: 0, y1: baseY, x2: plotW, y2: baseY, stroke: "var(--muted)", "stroke-width": 1 }, grid);
    el("line", { x1: AXIS_W - 1, y1: TOP, x2: AXIS_W - 1, y2: baseY, stroke: "var(--muted)", "stroke-width": 1 }, axisSvg);
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": LABEL_SIZE, "font-weight": 650,
      fill: "var(--ink)", transform: `translate(15 ${TOP + PLOT_H / 2}) rotate(-90)`,
    }, axisSvg).textContent = m.axisLabel;

    // -- hover band behind the marks (highlights the whole column group)
    const band = el("rect", {
      x: 0, y: TOP, width: slot, height: PLOT_H + 6, fill: "var(--ink)", opacity: 0, "pointer-events": "none",
    }, svg);

    const defs = el("defs", {}, svg);
    // -- imputed hatch, reused by every partially imputed bar. Crossed diagonals,
    //    matching the "x" hatch matplotlib draws in the static figure; the
    //    diagonals meet the tile corners exactly, so the grid tiles seamlessly.
    const pat = el("pattern", {
      id: "imp-hatch", width: 8, height: 8, patternUnits: "userSpaceOnUse",
    }, defs);
    el("path", {
      d: "M0,0 L8,8 M8,0 L0,8", stroke: "var(--paper)", "stroke-width": 1.5,
      opacity: 0.85, fill: "none",
    }, pat);

    // -- Everything data-bearing is clipped to the plot area. The axis floor is
    //    set from the bar values (a couple of very wide intervals would
    //    otherwise deflate the whole scale), so a long lower CI whisker can
    //    reach past it and run into the method names below. Clip it there; the
    //    exact interval stays in the tooltip and the data table.
    const clip = el("clipPath", { id: "plot-clip" }, defs);
    el("rect", { x: 0, y: 0, width: plotW, height: baseY }, clip);

    // -- bars: one concentric group per method
    const barsG = el("g", { "clip-path": "url(#plot-clip)" }, svg);
    entries.forEach((entry, i) => {
      const cx = i * slot + slot / 2;
      // Widest bar first, narrowest last: the bars are concentric, so painting
      // them by width guarantees each one stays visible. Ordering by height
      // instead loses a variant outright whenever a narrower bar is the taller
      // of the two (TabSTAR's tuned bar sat 1 Elo above tuned + ensembled, and
      // the wider bar covered it completely).
      const relOf = p => (VARIANT_STYLE[p.variant] || VARIANT_STYLE["Default"]).rel;
      const drawn = entry.points
        .filter(p => state.variants.has(p.variant) && p[m.key] != null)
        .slice()
        .sort((a, b) => relOf(b) - relOf(a));
      for (const p of drawn) {
        const style = VARIANT_STYLE[p.variant] || VARIANT_STYLE["Default"];
        const w = barUnit * style.rel;
        const top = Y(p[m.key]);
        const rect = {
          x: cx - w / 2, y: Math.min(top, baseY), width: w, height: Math.max(1, Math.abs(baseY - top)),
        };
        // 2px surface ring, not a border: it is what keeps a nested bar legible
        // against the wider bar it sits inside.
        el("rect", { ...rect, fill: style.color, stroke: "var(--paper)", "stroke-width": 1.5, rx: 2 }, barsG);
        if (p.imputed) el("rect", { ...rect, fill: "url(#imp-hatch)", rx: 2 }, barsG);
        if (m.ci && p[m.ci.lo] != null) {
          const cap = Math.max(2.5, w * 0.34);
          const whisk = el("g", {
            stroke: `color-mix(in srgb, ${style.color} 55%, var(--ink))`, "stroke-width": 1.4, opacity: 0.9,
          }, barsG);
          el("line", { x1: cx, y1: Y(p[m.ci.lo]), x2: cx, y2: Y(p[m.ci.hi]) }, whisk);
          el("line", { x1: cx - cap, y1: Y(p[m.ci.hi]), x2: cx + cap, y2: Y(p[m.ci.hi]) }, whisk);
          el("line", { x1: cx - cap, y1: Y(p[m.ci.lo]), x2: cx + cap, y2: Y(p[m.ci.lo]) }, whisk);
        }
      }
    });

    // -- x axis: the method name, colored by model family (the legend below
    //    names the colors).
    const xg = el("g", {}, svg);
    // Hairlines share one group that is painted before every name, so a name
    // nudged sideways at the edge (see below) passes over a neighbouring
    // column's hairline instead of being crossed out by it.
    const xLines = el("g", {}, xg);
    const xLabels = el("g", {}, xg);
    entries.forEach((entry, i) => {
      const cx = i * slot + slot / 2;
      const row = labelRows === 1 ? 0 : i % 2;
      const y = baseY + LABEL_TOP + row * LABEL_ROW;
      // A hairline drops the staggered row back to its own column.
      if (row) {
        el("line", {
          x1: cx, y1: baseY + 3, x2: cx, y2: y - 10,
          stroke: "var(--line)", "stroke-width": 1,
        }, xLines);
      }
      const t = el("text", {
        x: cx, y, "font-size": LABEL_SIZE, "text-anchor": "middle", "font-weight": 650,
        fill: FAM_INK[entry.family] || "var(--ink)",
        // A halo in the surface color keeps the glyphs legible where a nudged
        // name crosses a hairline, rather than the line running through them.
        "paint-order": "stroke", stroke: "var(--paper)", "stroke-width": 3,
      }, xLabels);
      t.textContent = labels[i];
      fitLabel(t, labels[i], labelWidths[i], slot * labelRows - 8);
      // The outermost columns sit only half a slot from the edge, so a name
      // wider than one slot would reach past the SVG viewport and be cut off
      // there (a name may occupy two slots when the rows are staggered). Nudge
      // it inwards by just enough to stay whole; every other label keeps its
      // column centre, and the bar below still marks the column.
      const half = t.getComputedTextLength() / 2;
      t.setAttribute("x", Math.max(half + 1, Math.min(cx, plotW - half - 1)));
    });

    // -- reference pipelines as threshold lines. Their names live in the legend
    //    (matched by dash pattern) rather than on the line, where they would
    //    cover the tallest bars at every scroll position.
    const refG = el("g", {}, svg);
    const tagYs = [];
    shownRefs.forEach((r, i) => {
      const ry = Y(r[state.metric]);
      if (ry < TOP || ry > baseY) return;
      el("line", {
        x1: 0, y1: ry, x2: plotW, y2: ry, stroke: "var(--fam-reference)", "stroke-width": 1.8,
        "stroke-dasharray": REF_DASHES[i % REF_DASHES.length], opacity: 0.95,
      }, refG);
      // Sticky value tag in the axis pane, so the threshold stays readable at
      // any scroll position. Two nearby thresholds would print on top of each
      // other, so nudge each tag clear of the ones already placed; a tag wins
      // over a tick label it would sit on.
      let ty = ry;
      while (tagYs.some(y => Math.abs(y - ty) < 12)) ty += 12;
      tagYs.push(ty);
      for (const t of tickLabels) {
        if (Math.abs(t.y - ty) < 10) t.node.remove();
      }
      el("text", {
        x: AXIS_W - 10, y: ty + 4, "text-anchor": "end", "font-size": 11, "font-weight": 650,
        fill: "var(--fam-reference)",
      }, axisSvg).textContent = fmtMetric(m, r[state.metric]);
    });
    buildLegend(m, shownRefs);

    // -- hit targets: one full-height column per method (>= 34px wide)
    const hits = el("g", {}, svg);
    entries.forEach((entry, i) => {
      const h = el("rect", {
        x: i * slot, y: TOP, width: slot, height: PLOT_H + 6, fill: "transparent", cursor: "pointer",
      }, hits);
      h.addEventListener("mouseenter", ev => {
        band.setAttribute("x", i * slot);
        band.setAttribute("opacity", 0.06);
        showTip(entry, ev);
      });
      h.addEventListener("mousemove", ev => tip.move(ev));
      h.addEventListener("mouseleave", () => { band.setAttribute("opacity", 0); tip.hide(); });
      h.addEventListener("click", () => toggleMethod(entry.method));
    });

    postHeight();
  }

  function showTip(entry, ev) {
    const m = metric();
    let html = `<div class="t-name">${entry.method}</div><div>${entry.family}</div>`;
    for (const p of entry.points) {
      if (!state.variants.has(p.variant) || p[m.key] == null) continue;
      const ci = m.ci && p[m.ci.lo] != null
        ? ` <span class="t-var">(${fmtNum(p[m.ci.lo], m.decimals)}–${fmtNum(p[m.ci.hi], m.decimals)})</span>`
        : "";
      html += `<div><span class="t-var">${p.variant}:</span> <b>${fmtMetric(m, p[m.key])}</b>${ci}</div>`;
    }
    if (entry.imputed) html += `<div class="t-imp">Imputed on ${fmtNum(entry.imputed_pct, 0)}% of datasets</div>`;
    tip.show(html, ev);
  }

  // ---------- chips ----------
  const chipsBox = document.getElementById("chips");
  const chipByMethod = new Map();
  const famChips = new Map();

  function familyMembers(fam) {
    const out = [...byMethod.values()].filter(e => e.family === fam).map(e => e.method);
    for (const r of refs) if (r.family === fam) out.push(r.method);
    return out;
  }
  function isOn(name) { return state.methods.has(name) || state.refs.has(name); }

  function buildChips() {
    const rankMetric = metricByKey[CONFIG.rankMetric] || METRICS[0];
    const head = document.createElement("div");
    head.className = "chips-head";
    head.textContent = "Methods shown — click to remove, click a family to toggle the whole group";
    chipsBox.appendChild(head);
    for (const fam of FAM_ORDER) {
      const members = familyMembers(fam);
      if (!members.length) continue;
      members.sort((a, b) => chipRank(a, rankMetric) - chipRank(b, rankMetric));
      const row = document.createElement("div");
      row.className = "chiprow";
      const famBtn = document.createElement("button");
      famBtn.className = "famchip";
      famBtn.style.setProperty("--fam", FAM_VAR[fam]);
      famBtn.innerHTML = famChipLabel(fam, members.length);
      famBtn.title = `Toggle all ${members.length} ${fam} methods`;
      famBtn.addEventListener("click", () => toggleFamily(fam));
      row.appendChild(famBtn);
      famChips.set(fam, famBtn);
      const set = document.createElement("div");
      set.className = "chipset";
      for (const name of members) {
        const entry = byMethod.get(name);
        const b = document.createElement("button");
        b.className = "chip";
        b.style.setProperty("--fam", FAM_VAR[fam]);
        b.appendChild(Object.assign(document.createElement("span"), { className: "dot" }));
        b.appendChild(Object.assign(document.createElement("span"), { textContent: name }));
        if (entry && entry.imputed) {
          const mark = document.createElement("span");
          mark.className = "imp-mark";
          mark.textContent = "‡";
          b.appendChild(mark);
        }
        b.title = name + (entry && entry.imputed ? " — partially imputed" : "");
        b.addEventListener("click", () => toggleMethod(name));
        set.appendChild(b);
        chipByMethod.set(name, b);
      }
      row.appendChild(set);
      chipsBox.appendChild(row);
    }
  }
  function chipRank(name, m) {
    const entry = byMethod.get(name);
    if (entry) return rankVal(entry, m);
    const r = refs.find(x => x.method === name);
    const v = r ? r[m.key] : null;
    return v == null ? Infinity : (m.lowerBetter ? v : -v);
  }
  function syncChips() {
    for (const [name, b] of chipByMethod) b.setAttribute("aria-pressed", String(isOn(name)));
    for (const [fam, b] of famChips) b.setAttribute("aria-pressed", String(familyMembers(fam).every(isOn)));
  }

  function toggleMethod(name) {
    const set = byMethod.has(name) ? state.methods : state.refs;
    if (set.has(name)) set.delete(name); else set.add(name);
    syncChips();
    render();
  }
  function toggleFamily(fam) {
    const members = familyMembers(fam);
    const allOn = members.every(isOn);
    for (const name of members) {
      const set = byMethod.has(name) ? state.methods : state.refs;
      if (allOn) set.delete(name); else set.add(name);
    }
    syncChips();
    render();
  }
  function setMethods(names) {
    state.methods = new Set(names);
    syncChips();
    render();
  }

  document.getElementById("btn-all").addEventListener("click", () => {
    state.refs = new Set(refs.map(r => r.method));
    setMethods(byMethod.keys());
  });
  document.getElementById("btn-none").addEventListener("click", () => {
    state.refs = new Set();
    setMethods([]);
  });
  document.getElementById("btn-top").addEventListener("click", () => {
    const m = metric();
    // Rank explicitly rather than reusing the display sort: which 15 methods are
    // kept must not depend on which end of the axis the best ones are drawn at
    // (sorting "best on the right" would otherwise select the 15 worst).
    const top = [...byMethod.values()]
      .sort((a, b) => rankVal(a, m) - rankVal(b, m))
      .slice(0, 15)
      .map(e => e.method);
    state.refs = new Set(refs.map(r => r.method));
    setMethods(top);
  });

  // ---------- selectors ----------
  yminRange.addEventListener("input", ev => {
    state.yMin = Number(ev.target.value);
    render();
  });

  const metricSelect = document.getElementById("metric-select");
  for (const m of METRICS) {
    metricSelect.appendChild(Object.assign(document.createElement("option"), { value: m.key, textContent: m.label }));
  }
  metricSelect.addEventListener("change", ev => {
    state.metric = ev.target.value;
    state.yMin = null;  // the previous floor means nothing on a new scale
    buildTable();
    render();
  });

  const sortSelect = document.getElementById("sort-select");
  for (const [value, label] of [
    ["best", "Best on the left"], ["worst", "Best on the right"],
    ["family", "Model family"], ["name", "A–Z"],
  ]) {
    sortSelect.appendChild(Object.assign(document.createElement("option"), { value, textContent: label }));
  }
  sortSelect.value = state.sort;
  sortSelect.addEventListener("change", ev => { state.sort = ev.target.value; render(); });

  // Variant toggles: buttons rather than chips, since they filter the series
  // rather than the rows. "Default" gets no button at all — every method has a
  // default result, so it is the baseline of the chart rather than an option
  // (switching it off would leave the default-only methods with no bar). Only
  // the extras layered on top, tuning and ensembling, are toggleable; the legend
  // still carries Default's color.
  const ALWAYS_SHOWN = "Default";
  const variantBtns = document.getElementById("variant-btns");
  const variantBtnByKey = new Map();
  for (const v of VARIANT_ORDER) {
    if (v === ALWAYS_SHOWN || !POINTS.some(p => p.variant === v)) continue;
    const b = document.createElement("button");
    b.className = "btn";
    b.textContent = v === "Tuned + Ens." ? "Tuned + Ensembled" : v;
    b.style.setProperty("--fam", VARIANT_STYLE[v].color);
    b.title = `Show or hide the ${b.textContent.toLowerCase()} bars`;
    b.addEventListener("click", () => {
      if (state.variants.has(v)) state.variants.delete(v);
      else state.variants.add(v);
      syncVariantBtns();
      render();
    });
    variantBtns.appendChild(b);
    variantBtnByKey.set(v, b);
  }
  function syncVariantBtns() {
    for (const [v, b] of variantBtnByKey) {
      const on = state.variants.has(v);
      b.setAttribute("aria-pressed", String(on));
      b.style.opacity = on ? "1" : "0.45";
      b.style.borderColor = on ? VARIANT_STYLE[v].color : "var(--line)";
    }
  }

  // ---------- legend ----------
  // Rebuilt on every render: it names the reference lines (each by its dash
  // pattern and current value), which change with the metric and the selection.
  function buildLegend(m, shownRefs) {
    const parts = [];
    for (const v of VARIANT_ORDER) {
      if (!POINTS.some(p => p.variant === v)) continue;
      const label = v === "Tuned + Ens." ? "Tuned + Ensembled" : v;
      const off = state.variants.has(v) ? "" : ' style="opacity:0.4"';
      parts.push(
        `<span class="item"${off}><svg width="12" height="12" viewBox="0 0 12 12">` +
        `<rect x="1" y="1" width="10" height="10" rx="2" fill="${VARIANT_STYLE[v].color}"/></svg> ${label}</span>`);
    }
    if (m.ci) {
      parts.push('<span class="item"><svg width="12" height="14" viewBox="0 0 12 14">' +
        '<path d="M6,2 V12 M2,2 H10 M2,12 H10" stroke="var(--muted)" stroke-width="1.4" fill="none"/></svg> 95% CI</span>');
    }
    shownRefs.forEach((r, i) => {
      parts.push(`<span class="item"><svg width="26" height="8" viewBox="0 0 26 8">` +
        `<line x1="0" y1="4" x2="26" y2="4" stroke="var(--fam-reference)" stroke-width="1.8" ` +
        `stroke-dasharray="${REF_DASHES[i % REF_DASHES.length]}"/></svg> ${r.method} · ${fmtMetric(m, r[state.metric])}</span>`);
    });
    if (POINTS.some(p => p.imputed)) {
      parts.push('<span class="item"><svg width="14" height="14" viewBox="0 0 14 14">' +
        '<rect x="1" y="1" width="12" height="12" rx="2" fill="var(--pt-muted)"/>' +
        '<path d="M1,1 L7,7 M7,1 L1,7 M7,7 L13,13 M13,7 L7,13 M1,7 L7,13 M7,7 L13,1" ' +
        'stroke="var(--paper)" stroke-width="1.5" fill="none"/>' +
        "</svg> &Dagger; partially imputed</span>");
    }
    // Model family, named rather than merely pointed at: without the colors
    // spelled out, the swatch under each column decodes to nothing.
    const families = FAM_ORDER.filter(f => POINTS.some(p => !p.reference && p.family === f));
    if (families.length > 1) {
      parts.push('<span class="legendbreak"></span><span class="item">Family:</span>');
      for (const fam of families) {
        parts.push(`<span class="item"><svg width="16" height="9" viewBox="0 0 16 9">` +
          `<rect x="0" y="1" width="16" height="7" rx="2" fill="${FAM_VAR[fam]}"/></svg> ` +
          `<span style="color:${FAM_INK[fam]}">${fam}</span></span>`);
      }
    }
    document.getElementById("legendstrip").innerHTML = parts.join("");
  }

  // ---------- data table (the WCAG-clean twin of the chart) ----------
  function buildTable() {
    const m = metric();
    const rows = [...POINTS].filter(p => p[m.key] != null).sort((a, b) =>
      m.lowerBetter ? a[m.key] - b[m.key] : b[m.key] - a[m.key]);
    let html = "<table><thead><tr><th>Method</th><th>Variant</th><th>Family</th>";
    for (const x of METRICS) html += `<th>${x.label}</th>`;
    html += "<th>Imputed</th></tr></thead><tbody>";
    for (const p of rows) {
      html += `<tr><td>${p.method}</td><td>${p.variant || "—"}</td><td>${p.family}</td>`;
      for (const x of METRICS) html += `<td>${fmtMetric(x, p[x.key])}</td>`;
      html += `<td>${p.imputed ? fmtNum(p.imputed_pct, 0) + "%" : "—"}</td></tr>`;
    }
    document.getElementById("tblwrap").innerHTML = html + "</tbody></table>";
  }

  // ---------- paper view ----------
  setUpPaperView(render);
  // The y-axis lives in its own pane, so it is offset back into place.
  setUpExport(() => [{ svg: axisSvg, dx: 0 }, { svg: svg, dx: AXIS_W }], () => slugify(document.title));

  // ---------- boot ----------
  document.querySelector("details.datatable").addEventListener("toggle", postHeight);
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 120);
  });

  buildChips();
  buildTable();
  syncChips();
  syncVariantBtns();
  render();  // also builds the legend (it depends on the metric + selection)
})();
</script>
</body>
</html>
"""
