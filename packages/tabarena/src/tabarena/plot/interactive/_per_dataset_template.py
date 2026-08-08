"""HTML/CSS/JS template for the self-contained per-dataset browser.

A master-detail page: a sortable, filterable list of datasets on top, and the selected
dataset's detail below it. Each row carries a strip showing where every method landed on that
dataset, with a star on the contender the reader picked; the detail below shows that dataset's
tuning trajectories and its full ranking.

The detail sits in a fixed place under the list rather than expanding inside the row, so
stepping through datasets with the arrow keys updates the chart without the page moving under
the reader.

Kept as a Python string constant (rather than a package-data file) so it ships with the package
without any build-system data-file configuration.
:func:`tabarena.plot.interactive.per_dataset_explorer.build_per_dataset_explorer_html`
substitutes the placeholders (see
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

PER_DATASET_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  /* The contender's mark is one fixed colour whoever the contender is. Taking its family hue
     made it compete with the family-coloured field it exists to stand out from — a green star
     among green tree-based dots — and it had to be re-learned on every change of contender.

     Yellow rather than a softer gold: the nearest family colour is the System orange (#f0a35a),
     and an amber star sat close enough to it to reintroduce the same confusion. This is well
     clear of every family hue, and no family owns yellow. Declared across the same four scopes
     as the shared tokens (see _explorer_shared): base is light, the media query follows the OS,
     and the two data-theme stamps let an embedding page force either one. */
  :root { --contender: #cf9200; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --contender: #ffd21e; }
  }
  :root[data-theme="dark"] { --contender: #ffd21e; }
  :root[data-theme="light"] { --contender: #cf9200; }

  .pd-page { position: relative; }

  .pd-controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-bottom: 8px; }
  .pd-search {
    font: 500 12.5px/1.2 system-ui, sans-serif; color: var(--ink); background: var(--chip-bg);
    border: 1px solid var(--line); border-radius: 7px; padding: 6px 9px; min-width: 200px; flex: 1 1 200px;
    max-width: 320px;
  }
  .pd-search:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .pd-filtergroup { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px; }
  .pd-filtergroup .grouplabel {
    font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; margin-right: 2px;
  }
  .pd-fchip {
    font: 600 11.5px/1 system-ui, sans-serif; color: var(--muted); background: none;
    border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; cursor: pointer;
  }
  .pd-fchip:hover { border-color: var(--muted); color: var(--ink); }
  .pd-fchip[aria-pressed="true"] {
    border-color: var(--accent); color: var(--ink);
    background: color-mix(in srgb, var(--accent) 15%, transparent);
  }

  /* --- The dataset list -----------------------------------------------------
     Its own scroll box: the page is embedded in an iframe sized to its content, so a
     51-row table left to grow would make the frame taller than the screen and put the
     detail chart out of sight. A bounded list also lets the arrow keys scroll the
     selected row into view, which a scripted scroll of the host page cannot do. */
  /* Resizable, because the list and the detail chart together are taller than a laptop screen
     and which of the two you want more of depends on what you are doing. `height` rather than
     `max-height`: a max would clamp the resize and leave the box ungrowable. */
  .pd-listbox { position: relative; }
  .pd-listwrap {
    overflow: auto; height: 340px; min-height: 92px;
    border: 1px solid var(--line); border-radius: 10px;
    scrollbar-color: var(--muted) transparent;
  }
  .pd-listwrap::-webkit-scrollbar { width: 12px; height: 12px; }
  .pd-listwrap::-webkit-scrollbar-thumb {
    background: var(--muted); border-radius: 8px; border: 3px solid transparent; background-clip: content-box;
  }
  .pd-listwrap::-webkit-scrollbar-thumb:hover { background: var(--ink); background-clip: content-box; }
  /* "There is more" has to be visible without scrolling first, so the cut-off rows are faded
     under a label rather than left to look like the end of the list. */
  .pd-more {
    position: absolute; left: 1px; right: 13px; bottom: 1px; height: 40px; pointer-events: none;
    display: none; align-items: flex-end; justify-content: center; padding-bottom: 3px;
    font: 650 10.5px/1 system-ui, sans-serif; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--muted); border-radius: 0 0 9px 9px;
    background: linear-gradient(to top, var(--paper) 38%, transparent);
  }
  .pd-listbox.has-more .pd-more { display: flex; }

  /* An explicit handle: the browser's own resize grip is a few pixels in one corner and nobody
     finds it. This one spans the list, says what it does, and takes the arrow keys. */
  .pd-griprow { display: flex; align-items: center; gap: 10px; margin-top: 5px; }
  .pd-grip {
    flex: 1 1 auto; display: flex; align-items: center; justify-content: center; gap: 9px;
    cursor: ns-resize; user-select: none; touch-action: none; padding: 3px 8px; border-radius: 7px;
    font: 600 10.5px/1 system-ui, sans-serif; letter-spacing: 0.07em; text-transform: uppercase;
    color: var(--muted); background: var(--chip-bg); border: 1px solid var(--line);
  }
  .pd-grip:hover, .pd-grip.is-dragging { color: var(--ink); border-color: var(--muted); }
  .pd-grip:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .pd-gripdots {
    flex: 1 1 auto; height: 0; border-top: 2px dotted currentColor; opacity: 0.55; max-width: 220px;
  }
  .pd-listwrap:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  table.pd-table {
    border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums;
  }
  table.pd-table thead th {
    position: sticky; top: 0; z-index: 3; background: var(--card); color: var(--muted);
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.05em; text-transform: uppercase;
    text-align: left; padding: 7px 9px; white-space: nowrap; cursor: pointer; user-select: none;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  table.pd-table thead th:hover { color: var(--accent); }
  table.pd-table thead th.sortable::after { content: "\2195"; font-size: 0.8em; opacity: 0.35; margin-left: 4px; }
  table.pd-table thead th[aria-sort="ascending"]::after { content: "\25B2"; opacity: 0.9; }
  table.pd-table thead th[aria-sort="descending"]::after { content: "\25BC"; opacity: 0.9; }
  table.pd-table td {
    padding: 4px 9px; border-bottom: 1px solid var(--line); white-space: nowrap; vertical-align: middle;
  }
  table.pd-table tbody tr { cursor: pointer; }
  table.pd-table tbody tr:hover td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
  table.pd-table tbody tr.sel td { background: color-mix(in srgb, var(--accent) 17%, transparent); }
  table.pd-table tbody tr.sel td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  td.pd-num, th.pd-num { text-align: right; }
  .pd-pos { color: var(--muted); }
  .pd-name { font-weight: 620; }
  .pd-tag {
    display: inline-block; font-size: 10.5px; font-weight: 650; letter-spacing: 0.03em;
    border: 1px solid var(--line); border-radius: 5px; padding: 1px 5px; color: var(--muted);
  }
  /* The variant badge, coloured with the same tokens the charts plot the variants in. */
  .pd-var {
    display: inline-block; font: 700 9.5px/1.5 system-ui, sans-serif; letter-spacing: 0.04em;
    border-radius: 4px; padding: 0 4px; margin-left: 5px; color: #14161a; background: var(--vc);
  }
  .pd-stripcell { width: 250px; padding-top: 0; padding-bottom: 0; }
  /* A system carries its configuration in its name, so the winner cell is bounded and clipped
     rather than allowed to push the table wider than the frame. */
  table.pd-table td:last-child { max-width: 190px; overflow: hidden; text-overflow: ellipsis; }
  .pd-stripsvg { display: block; }
  .pd-hint { font-size: 12px; color: var(--muted); margin: 0 2px 6px; }
  .pd-hint kbd {
    font: 600 11px/1 system-ui, sans-serif; border: 1px solid var(--line); border-bottom-width: 2px;
    border-radius: 4px; padding: 2px 5px; background: var(--chip-bg);
  }

  /* --- The detail pane ------------------------------------------------------ */
  .pd-detail { margin-top: 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); }
  /* What the dataset *is*, reading as a caption under the chart it describes. The tinted panel
     is what makes it a block rather than a line of grey text the eye slides past; the facts
     themselves stay a plain run-on list, which packs far more of them onto a line than pills do.
     The name leads it, and there is no separate header above the chart: the selected row in the
     table is already highlighted, so a second copy of the name up there said nothing new. */
  .pd-about {
    margin-top: 12px; padding: 8px 11px 9px; border-radius: 9px;
    background: color-mix(in srgb, var(--accent) 8%, transparent);
    border: 1px solid color-mix(in srgb, var(--accent) 26%, var(--line));
  }
  .pd-aboutname { font-size: 13.5px; font-weight: 700; color: var(--ink); }
  .pd-aboutfacts {
    display: flex; flex-wrap: wrap; gap: 2px 10px; margin-top: 3px;
    font-size: 11.5px; color: var(--muted);
  }
  .pd-aboutfacts .item { white-space: nowrap; }
  .pd-aboutfacts .item b { color: var(--ink); font-weight: 620; font-variant-numeric: tabular-nums; }
  .pd-dbody { display: flex; gap: 14px; align-items: flex-start; padding: 10px 12px 12px; }
  .pd-chartcol { flex: 1 1 auto; min-width: 0; }
  .pd-rankcol { flex: 0 0 268px; min-width: 220px; }
  @media (max-width: 820px) {
    .pd-dbody { flex-wrap: wrap; }
    .pd-rankcol { flex: 1 1 100%; }
  }
  .pd-chartbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 12px; margin-bottom: 6px; }
  /* `.hint` is only styled inside `.controls` in the shared sheet; these two bars are the
     per-dataset page's own. */
  .pd-chartbar .hint, .pd-controls .hint { font-size: 12.5px; color: var(--muted); }
  .pd-empty { color: var(--muted); font-size: 12.5px; font-style: italic; padding: 26px 4px; }
  .legendstrip .legendbreak { flex-basis: 100%; height: 0; }

  .pd-ranklist {
    list-style: none; margin: 0; padding: 0; overflow: auto; max-height: 372px;
    scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent;
  }
  .pd-ranklist::-webkit-scrollbar { width: 9px; }
  .pd-ranklist::-webkit-scrollbar-thumb {
    background: var(--pt-muted); border-radius: 8px; border: 3px solid transparent; background-clip: content-box;
  }
  .pd-rankrow {
    display: grid; grid-template-columns: 20px 1fr auto; align-items: center; gap: 6px;
    padding: 3px 5px; border-radius: 6px; cursor: pointer; font-size: 12px;
  }
  .pd-rankrow:hover { background: color-mix(in srgb, var(--accent) 9%, transparent); }
  .pd-rankrow.is-contender {
    background: color-mix(in srgb, var(--contender) 20%, transparent); font-weight: 650;
  }
  .pd-rankrow.is-contender .star { color: var(--contender); }
  .pd-rankrow .pos { color: var(--muted); text-align: right; font-variant-numeric: tabular-nums; }
  .pd-rankrow .who { display: flex; align-items: center; gap: 5px; min-width: 0; }
  .pd-rankrow .who .nm { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
  .pd-rankrow .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--fam); flex: none; }
  .pd-rankrow .val { font-variant-numeric: tabular-nums; color: var(--muted); }
  .pd-rankrow.is-active .nm { text-decoration: underline; text-decoration-color: var(--fam); text-underline-offset: 3px; }
  .pd-rankbar { grid-column: 1 / -1; height: 3px; border-radius: 2px; background: var(--line); overflow: hidden; }
  .pd-rankbar span { display: block; height: 100%; background: var(--fam); opacity: 0.7; }

  .chartbox { position: relative; }
  .chartbox svg { display: block; }
  body.paper .pd-controls,
  body.paper .pd-listbox,
  body.paper .pd-griprow,
  body.paper .pd-listwrap,
  body.paper .pd-hint,
  body.paper .pd-rankcol { display: none !important; }
  body.paper .pd-detail { border: none; background: none; }
</style>
</head>
<body>
<div class="pd-page" id="page">
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, the selected dataset's chart only">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>

  <div class="pd-controls">
    <label class="metricpick">Contender
      <select id="contender" title="The method the star marks, and the one the ranking columns report"></select>
    </label>
    <input class="pd-search" id="search" type="search"
           placeholder="Filter datasets by name, domain or source" aria-label="Filter datasets">
    <div class="pd-filtergroup" id="taskfilter"></div>
    <div class="pd-filtergroup" id="sizefilter"></div>
    <span class="hint" id="count"></span>
  </div>

  <div class="exportbar" id="exportbar" hidden>
    <span class="hint">Export the chart</span>
    <button class="btn" id="btn-svg" title="Download as SVG — vector, keeps text selectable">SVG</button>
    <button class="btn" id="btn-pdf" title="Download as a one-page PDF">PDF</button>
    <button class="btn" id="btn-png" title="Download as PNG at 3x scale">PNG</button>
  </div>

  <p class="pd-hint" id="listhint"></p>
  <div class="pd-listbox" id="listbox">
    <div class="pd-listwrap" id="listwrap" tabindex="0" aria-label="Datasets">
      <table class="pd-table">
        <thead id="thead"></thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
    <div class="pd-more" id="listmore" aria-hidden="true">more below &#9662;</div>
  </div>
  <div class="pd-griprow">
    <div class="pd-grip" id="listgrip" role="separator" aria-orientation="horizontal" tabindex="0"
         title="Drag up or down to resize the list (or use the arrow keys)">
      <span class="pd-gripdots"></span><span>drag to resize the list</span><span class="pd-gripdots"></span>
    </div>
    <button class="btn" id="btn-fit" aria-pressed="false"
            title="Size the list so it and the chart below are on screen together">&#8597; Fit table and figure to screen</button>
  </div>

  <div class="pd-detail" id="detail"></div>
  <div class="tooltip"></div>
</div>

<script>
(function () {
  "use strict";
  const CONFIG = __CONFIG_JSON__;
  const POINTS = __POINTS_JSON__;

__BASE_JS__

  const DATASETS = CONFIG.datasets;
  const METHODS = CONFIG.methods;
  // Baseline and Other share a bucket everywhere else on the site; normalize once so every
  // colour, legend and chip lookup below sees the merged family.
  for (const m of METHODS) m.family = famOf(m.family);
  const TRAJ_METHODS = CONFIG.trajectoryMethods || [];
  const TRAJ = (CONFIG.trajectory && CONFIG.trajectory.rows) || [];
  const METRIC_DISPLAY = CONFIG.metricDisplay || {};
  const SIZE_BUCKETS = CONFIG.sizeBuckets || [];

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;

  const page = document.getElementById("page");
  const tip = makeTooltip(page);
  const listWrap = document.getElementById("listwrap");
  const tbody = document.getElementById("tbody");
  const detailEl = document.getElementById("detail");

  // ---------- indexes ----------
  const nD = DATASETS.length;
  const byDataset = Array.from({ length: nD }, () => []);
  const cell = new Map();                       // "dataset:method" -> the point
  for (const p of POINTS) {
    byDataset[p.d].push(p);
    cell.set(p.d + ":" + p.m, p);
  }
  const key = (d, m) => cell.get(d + ":" + m);

  // Per-dataset trajectories, one polyline per method, ordered by tuning budget.
  const trajByDataset = Array.from({ length: nD }, () => new Map());
  for (const row of TRAJ) {
    const [d, m, n, x, e, i] = row;
    let arr = trajByDataset[d].get(m);
    if (!arr) { arr = []; trajByDataset[d].set(m, arr); }
    arr.push({ n: n, x: x, e: e, i: i });
  }
  for (const perMethod of trajByDataset) {
    for (const arr of perMethod.values()) arr.sort((a, b) => (a.n || 0) - (b.n || 0));
  }
  // The trajectory frame is keyed on a method's name without its tuning variant, so a
  // contender picked as "CatBoost (tuned)" finds the CatBoost line.
  const trajIndexByName = new Map(TRAJ_METHODS.map((name, i) => [name, i]));

  function quantile(sorted, q) {
    if (!sorted.length) return 0;
    const pos = (sorted.length - 1) * q;
    const lo = Math.floor(pos), hi = Math.ceil(pos);
    return sorted[lo] + (sorted[hi] - sorted[lo]) * (pos - lo);
  }

  // Per-dataset summaries the list needs: how far the field spreads out, who won, and where
  // the middle of the field sits. `spread` is the 90th percentile of the gap to the best,
  // which is what the "field" column sorts on: the worst method alone would make that column
  // report how bad someone's outlier was rather than how much the choice matters here.
  const STATS = DATASETS.map((ds, d) => {
    const rows = byDataset[d];
    const imps = rows.map(p => p.i).filter(v => v != null).sort((a, b) => a - b);
    let best = null;
    for (const p of rows) if (p.e != null && (best === null || p.e < best.e)) best = p;
    return {
      spread: quantile(imps, 0.9),
      worst: imps.length ? imps[imps.length - 1] : 0,
      median: quantile(imps, 0.5),
      best: best,
      n: rows.length,
    };
  });

  // What a model on this dataset actually fits: the largest training split, which is well below
  // the dataset's own size (a 150,000-row dataset trains on 100,000). The list, the size filter
  // and the detail pane all read it from here so they cannot disagree.
  function trainRows(ds) { return ds.train_rows != null ? ds.train_rows : ds.rows; }

  function sizeKeyOf(ds) {
    const rows = trainRows(ds);
    if (rows == null) return null;
    for (const b of SIZE_BUCKETS) if (b.max == null || rows <= b.max) return b.key;
    return SIZE_BUCKETS.length ? SIZE_BUCKETS[SIZE_BUCKETS.length - 1].key : null;
  }
  const SIZE_OF = DATASETS.map(sizeKeyOf);

  // ---------- state ----------
  const state = {
    contender: Math.min(CONFIG.defaultContender || 0, METHODS.length - 1),
    query: "",
    task: "all",
    size: "all",
    sort: "rank",
    dir: 1,
    selected: -1,
    // The dataset's own metric, which is what a reader looking at one dataset came for.
    // Improvability is the axis the aggregate figure needs to make datasets comparable; here
    // it is the second choice. Safe as a default only because the axis clips at the 90th
    // percentile — raw error is unbounded, and one collapsed model would otherwise flatten
    // every method worth comparing into a line along the bottom.
    yKey: "e",
    // Trajectory methods drawn in colour. Empty means "the front plus the contender", which is
    // recomputed per dataset; a non-empty set is the reader's own choice and is left alone.
    picked: new Set(),
  };

  // ---------- formatting ----------
  function fmtErr(v) {
    if (v == null || !isFinite(v)) return "—";
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(2);
    if (a >= 0.1) return v.toFixed(4);
    if (a === 0) return "0";
    return v.toPrecision(3);
  }
  function fmtInt(v) {
    if (v == null || !isFinite(v)) return "—";
    return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  const TASK_SHORT = { binary: "binary", multiclass: "multi", regression: "reg" };
  function metricName(ds) { return METRIC_DISPLAY[ds.metric] || ds.metric || "error"; }
  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  const VARIANT_SHORT = { "Default": "D", "Tuned": "T", "Tuned + Ens.": "T+E" };
  function variantBadge(method) {
    const short = VARIANT_SHORT[method.variant];
    if (!short) return "";
    return `<span class="pd-var" style="--vc:${VARIANT_VAR[method.variant]}">${short}</span>`;
  }
  function methodHtml(mi) {
    const m = METHODS[mi];
    return `<span style="color:${FAM_INK[m.family]}">${escapeHtml(m.base)}</span>` + variantBadge(m);
  }

  // ---------- marks ----------
  function starPath(cx, cy, r) {
    let d = "";
    for (let i = 0; i < 10; i++) {
      const rad = i % 2 ? r * 0.44 : r;
      const a = -Math.PI / 2 + i * Math.PI / 5;
      d += (i ? "L" : "M") + (cx + rad * Math.cos(a)).toFixed(2) + "," + (cy + rad * Math.sin(a)).toFixed(2);
    }
    return d + "Z";
  }
  // ---------- the strip ----------
  // One row of the list: every method's gap to the best on that dataset, with a star on the
  // contender. The winner needs no mark of its own — it is the method sitting at zero, which is
  // where the axis starts — and the Winner column names it.
  //
  // The axis is log(1 + gap). A linear one is unreadable here: on most datasets the field
  // bunches inside a few percent of the best while one or two collapsed models sit at 80%, so
  // linear puts everything that matters in the first two pixels. Clipping the scale instead
  // would hide exactly the failures this page is for. Log keeps every method on the strip and
  // spends the width where the comparisons are; exact numbers are one hover away.
  const STRIP_W = 240, STRIP_H = 19, STRIP_PAD = 8;
  function buildStrip(d) {
    const st = STATS[d];
    const x0 = STRIP_PAD, x1 = STRIP_W - STRIP_PAD;
    const scale = Math.log1p(Math.max(st.worst, 1));
    const at = v => x0 + Math.min(1, Math.log1p(Math.max(v, 0)) / scale) * (x1 - x0);
    const svg = document.createElementNS(NS, "svg");
    svg.setAttribute("width", STRIP_W);
    svg.setAttribute("height", STRIP_H);
    svg.setAttribute("class", "pd-stripsvg");
    const mid = STRIP_H / 2;
    el("line", {
      x1: x0, y1: mid, x2: x1, y2: mid, stroke: "var(--line)", "stroke-width": 1.5, "stroke-linecap": "round",
    }, svg);
    // The middle of the field, as a reference for reading where the star sits.
    el("line", {
      x1: at(st.median), y1: mid - 5, x2: at(st.median), y2: mid + 5,
      stroke: "var(--muted)", "stroke-width": 1, opacity: 0.5,
    }, svg);
    for (const p of byDataset[d]) {
      if (p.i == null || p.m === state.contender) continue;
      el("circle", {
        cx: at(p.i), cy: mid, r: 2.7, fill: FAM_VAR[METHODS[p.m].family], opacity: 0.6,
      }, svg);
    }
    const c = key(d, state.contender);
    if (c && c.i != null) {
      el("path", {
        d: starPath(at(c.i), mid, 6.4), fill: "var(--contender)",
        stroke: "var(--card)", "stroke-width": 1,
      }, svg);
    }
    svg.addEventListener("mousemove", ev => {
      const rect = svg.getBoundingClientRect();
      const mx = ev.clientX - rect.left;
      let nearest = null, dist = Infinity;
      for (const p of byDataset[d]) {
        if (p.i == null) continue;
        const dx = Math.abs(at(p.i) - mx);
        if (dx < dist) { dist = dx; nearest = p; }
      }
      if (nearest && dist < 11) tip.show(stripTip(d, nearest), ev); else tip.hide();
    });
    svg.addEventListener("mouseleave", () => tip.hide());
    return svg;
  }

  function stripTip(d, p) {
    const ds = DATASETS[d];
    return `<div class="t-name">${escapeHtml(METHODS[p.m].name)}</div>` +
      `<div>${METHODS[p.m].family}</div>` +
      `<div>${escapeHtml(metricName(ds))} error: <b>${fmtErr(p.e)}</b></div>` +
      `<div>Rank: <b>${fmtNum(p.r, 1)}</b> of ${STATS[d].n}</div>` +
      `<div>Behind the best by <b>${fmtNum(p.i, 1)}%</b></div>` +
      (p.t != null ? `<div>Mean fit: <b>${fmtTime(p.t)}</b></div>` : "") +
      (p.q ? '<div class="t-imp">Imputed on this dataset</div>' : "");
  }

  // ---------- the list ----------
  const COLUMNS = [
    { key: "pos", label: "#", cls: "pd-pos", sortable: false },
    { key: "name", label: "Dataset", hint: "Sort alphabetically" },
    { key: "task", label: "Task" },
    { key: "train_rows", label: "Train rows", num: true,
      hint: "Rows a model actually fits here — the largest training split, not the dataset's size" },
    { key: "features", label: "Feat.", num: true },
    { key: "rank", label: "Rank", num: true, hint: "The contender's mean rank on this dataset" },
    { key: "gap", label: "Gap", num: true, hint: "How much lower the best method's error is than the contender's" },
    { key: "spread", label: "The field", hint: "How far the field spreads out on this dataset" },
    { key: "winner", label: "Winner", hint: "The method with the lowest error here" },
  ];

  function sortValue(d, sortKey) {
    const ds = DATASETS[d];
    const c = key(d, state.contender);
    switch (sortKey) {
      case "name": return (ds.name || "").toLowerCase();
      case "task": return ds.task || "";
      case "train_rows": return trainRows(ds);
      case "features": return ds.features;
      case "rank": return c ? c.r : null;
      case "gap": return c ? c.i : null;
      case "spread": return STATS[d].spread;
      case "winner": return STATS[d].best ? METHODS[STATS[d].best.m].name.toLowerCase() : "";
      default: return null;
    }
  }

  function matchesQuery(ds) {
    if (!state.query) return true;
    const hay = [ds.name, ds.key, ds.domain, ds.source, ds.task].filter(Boolean).join(" ").toLowerCase();
    return hay.includes(state.query);
  }

  function visibleDatasets() {
    const out = [];
    for (let d = 0; d < nD; d++) {
      const ds = DATASETS[d];
      if (!matchesQuery(ds)) continue;
      if (state.task !== "all" && ds.task !== state.task) continue;
      if (state.size !== "all" && SIZE_OF[d] !== state.size) continue;
      out.push(d);
    }
    const dir = state.dir;
    const compare = (a, b, sortKey, direction) => {
      const va = sortValue(a, sortKey), vb = sortValue(b, sortKey);
      // Missing values sink to the bottom whichever way the column is sorted.
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string") return direction * va.localeCompare(vb);
      return direction * (va - vb);
    };
    // A strong contender is rank 1 on a good many datasets, so the primary column ties
    // constantly; break by the gap and then by name rather than leaving the order arbitrary.
    out.sort((a, b) =>
      compare(a, b, state.sort, dir) || compare(a, b, "gap", dir) || compare(a, b, "name", 1));
    return out;
  }

  let visible = [];

  // The box fits its rows, up to a ceiling the reader sets with the grip or the fit button.
  const LIST_DEFAULT = 340, LIST_MIN = 92;
  let listCeiling = LIST_DEFAULT;
  function sizeList() {
    // The table, not the box: `scrollHeight` is never smaller than the element's own height, so
    // once a height is set it can no longer report that the content has shrunk, and a list
    // filtered down to one row would keep its full box.
    const table = listWrap.querySelector("table");
    const fits = (table ? table.offsetHeight : 0) + 2;
    // Before the frame has been laid out that measures 0, which would collapse the box to its
    // minimum on the very first render and never recover. Leave the height alone until the rows
    // have a measurable size; `sizeList` is called again once layout has happened.
    if (fits < 20) return;
    listWrap.style.height = Math.round(Math.max(LIST_MIN, Math.min(listCeiling, fits))) + "px";
    syncScrollCue();
  }
  new ResizeObserver(postHeight).observe(listWrap);

  const listBox = document.getElementById("listbox");
  function syncScrollCue() {
    const hidden = listWrap.scrollHeight - listWrap.scrollTop - listWrap.clientHeight;
    listBox.classList.toggle("has-more", hidden > 2);
  }
  listWrap.addEventListener("scroll", syncScrollCue);

  // --- resizing -------------------------------------------------------------------------
  const grip = document.getElementById("listgrip");
  let gripFromY = 0, gripFromH = 0;
  function setCeiling(height) {
    listCeiling = Math.max(LIST_MIN, Math.round(height));
    sizeList();
    postHeight();
  }
  grip.addEventListener("pointerdown", ev => {
    grip.setPointerCapture(ev.pointerId);
    grip.classList.add("is-dragging");
    gripFromY = ev.clientY;
    gripFromH = listWrap.getBoundingClientRect().height;
    ev.preventDefault();
  });
  grip.addEventListener("pointermove", ev => {
    if (!grip.hasPointerCapture(ev.pointerId)) return;
    setCeiling(gripFromH + (ev.clientY - gripFromY));
    setFitPressed(false);
  });
  grip.addEventListener("pointerup", ev => {
    grip.releasePointerCapture(ev.pointerId);
    grip.classList.remove("is-dragging");
  });
  grip.addEventListener("keydown", ev => {
    const step = { ArrowUp: -40, ArrowDown: 40, PageUp: -160, PageDown: 160 }[ev.key];
    if (step === undefined) return;
    // The list's own arrow-key stepping is on `document`; this handle owns the keys while it
    // has focus.
    ev.preventDefault();
    ev.stopPropagation();
    setCeiling(listWrap.getBoundingClientRect().height + step);
    setFitPressed(false);
  });

  // --- fit to screen --------------------------------------------------------------------
  // How tall the reader's visible band actually is. Embedded, the host measures it and sends it
  // over (this frame is sized to its own content, so `innerHeight` here says nothing about the
  // screen). Standalone, `innerHeight` is exactly right.
  let hostViewport = 0;
  const fitBtn = document.getElementById("btn-fit");
  function setFitPressed(on) { fitBtn.setAttribute("aria-pressed", String(on)); }
  function fitToScreen() {
    const available = hostViewport || window.innerHeight;
    // Everything except the two things that give: the list and the chart. What is left is split
    // between them, the chart taking what it can up to its normal height and the list the rest.
    const others = document.body.offsetHeight - listWrap.getBoundingClientRect().height - chartHeight;
    const budget = Math.max(LIST_FIT_MIN + CHART_H_MIN, available - others - 16);
    chartHeight = Math.max(CHART_H_MIN, Math.min(CHART_H_DEFAULT, budget - LIST_FIT_MIN));
    drawChart();
    setCeiling(budget - chartHeight);
  }
  fitBtn.addEventListener("click", () => {
    const on = fitBtn.getAttribute("aria-pressed") !== "true";
    setFitPressed(on);
    if (on) {
      fitToScreen();
    } else {
      chartHeight = CHART_H_DEFAULT;
      drawChart();
      setCeiling(LIST_DEFAULT);
    }
  });

  function buildHead() {
    let html = "<tr>";
    for (const col of COLUMNS) {
      const sortable = col.sortable !== false;
      const cls = (col.num ? "pd-num " : "") + (sortable ? "sortable" : "");
      const title = col.hint ? ` title="${escapeHtml(col.hint)}"` : "";
      html += `<th class="${cls}" data-k="${col.key}"${title}>${col.label}</th>`;
    }
    document.getElementById("thead").innerHTML = html + "</tr>";
    for (const th of document.querySelectorAll("#thead th.sortable")) {
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        // A second click on the same column flips it; a new column starts in its natural
        // direction — names and ranks read best ascending, sizes and gaps descending.
        if (state.sort === k) state.dir = -state.dir;
        else { state.sort = k; state.dir = ["train_rows", "features", "spread", "gap"].includes(k) ? -1 : 1; }
        renderList();
      });
    }
  }
  function syncHead() {
    for (const th of document.querySelectorAll("#thead th")) {
      if (th.dataset.k === state.sort) th.setAttribute("aria-sort", state.dir > 0 ? "ascending" : "descending");
      else th.removeAttribute("aria-sort");
    }
  }

  function renderList() {
    visible = visibleDatasets();
    syncHead();
    const frag = document.createDocumentFragment();
    visible.forEach((d, i) => {
      const ds = DATASETS[d];
      const c = key(d, state.contender);
      const st = STATS[d];
      const tr = document.createElement("tr");
      tr.dataset.d = String(d);
      if (d === state.selected) tr.className = "sel";
      const meta = [ds.domain, ds.source, ds.year].filter(Boolean).join(" · ");
      tr.innerHTML =
        `<td class="pd-pos">${i + 1}</td>` +
        `<td><span class="pd-name" title="${escapeHtml(meta || ds.key)}">${escapeHtml(ds.name)}</span></td>` +
        `<td><span class="pd-tag">${escapeHtml(TASK_SHORT[ds.task] || ds.task || "?")}</span></td>` +
        `<td class="pd-num">${fmtInt(trainRows(ds))}</td>` +
        `<td class="pd-num">${fmtInt(ds.features)}</td>` +
        `<td class="pd-num">${c ? fmtNum(c.r, 1) : "—"}` +
        `<span class="pd-pos"> / ${st.n}</span></td>` +
        `<td class="pd-num">${c && c.i != null ? fmtNum(c.i, 1) + "%" : "—"}</td>` +
        `<td class="pd-stripcell"></td>` +
        `<td>${st.best ? methodHtml(st.best.m) : "—"}</td>`;
      tr.lastElementChild.previousElementSibling.appendChild(buildStrip(d));
      tr.addEventListener("click", () => select(d));
      frag.appendChild(tr);
    });
    tbody.textContent = "";
    tbody.appendChild(frag);
    const total = nD, shown = visible.length;
    document.getElementById("count").textContent =
      shown === total ? `${total} datasets` : `${shown} of ${total} datasets`;
    sizeList();
    if (visible.length && !visible.includes(state.selected)) select(visible[0], { scroll: false });
    else postHeight();
  }

  function select(d, options) {
    const opts = options || {};
    state.selected = d;
    state.picked = new Set();
    for (const tr of tbody.children) tr.classList.toggle("sel", Number(tr.dataset.d) === d);
    if (opts.scroll !== false) {
      const row = [...tbody.children].find(tr => Number(tr.dataset.d) === d);
      if (row) row.scrollIntoView({ block: "nearest" });
    }
    renderDetail();
  }

  function step(delta) {
    if (!visible.length) return;
    const at = visible.indexOf(state.selected);
    const next = at < 0 ? 0 : Math.min(visible.length - 1, Math.max(0, at + delta));
    select(visible[next]);
  }

  // ---------- the detail pane ----------
  const Y_METRICS = [
    { key: "e", label: "Test error", lowerBetter: true, fromZero: false },
    { key: "i", label: "Improvability (%)", lowerBetter: true, fromZero: true },
  ];

  function renderDetail() {
    const d = state.selected;
    if (d < 0 || d >= nD) { detailEl.innerHTML = ""; return; }
    const ds = DATASETS[d];
    const items = [
      ["Task", TASK_SHORT[ds.task] || ds.task],
      ["Metric", metricName(ds)],
      ["Train rows", fmtInt(trainRows(ds))],
      ["Dataset rows", fmtInt(ds.rows)],
      ["Features", fmtInt(ds.features)],
      ds.classes != null && ds.classes > 0 ? ["Classes", fmtInt(ds.classes)] : null,
      ["Splits", fmtInt(ds.splits)],
      ["Methods", fmtInt(STATS[d].n)],
      ds.domain ? ["Domain", ds.domain] : null,
      ds.source ? ["Source", ds.source] : null,
      ds.year ? ["Year", String(ds.year)] : null,
    ].filter(Boolean);
    const facts = items
      .map(([k, v]) => `<span class="item">${k}: <b>${escapeHtml(v)}</b></span>`)
      .join("");
    detailEl.innerHTML =
      '<div class="pd-dbody">' +
      '<div class="pd-chartcol">' +
      '<div class="pd-chartbar">' +
      '<label class="metricpick">Y-axis <select id="pd-yaxis"></select></label>' +
      '<div class="btnrow">' +
      '<button class="btn" id="pd-front">Front + contender</button>' +
      '<button class="btn" id="pd-all">All</button>' +
      "</div>" +
      '<span class="hint">Click a line, a point or a method on the right to hold it in colour</span>' +
      "</div>" +
      '<div class="legendstrip" id="legendstrip"></div>' +
      '<div class="chartbox" id="pd-chartbox"><svg id="pd-chart" role="img" aria-label="Tuning trajectories"></svg></div>' +
      `<div class="pd-about"><div class="pd-aboutname">${escapeHtml(ds.name)}</div>` +
      `<div class="pd-aboutfacts">${facts}</div></div>` +
      "</div>" +
      '<div class="pd-rankcol">' +
      `<div class="grouplabel">Every method on ${escapeHtml(ds.name)}</div>` +
      '<ol class="pd-ranklist" id="pd-rank"></ol>' +
      "</div></div>";

    const ySelect = document.getElementById("pd-yaxis");
    for (const m of Y_METRICS) {
      const opt = document.createElement("option");
      opt.value = m.key;
      opt.textContent = m.key === "e" ? `Test error (${metricName(ds)})` : m.label;
      ySelect.appendChild(opt);
    }
    ySelect.value = state.yKey;
    ySelect.addEventListener("change", ev => { state.yKey = ev.target.value; drawChart(); });
    document.getElementById("pd-front").addEventListener("click", () => { state.picked = new Set(); drawChart(); });
    document.getElementById("pd-all").addEventListener("click", () => {
      state.picked = new Set(trajByDataset[d].keys());
      drawChart();
    });
    renderRanking(d);
    buildLegend(d);
    drawChart();
  }

  function renderRanking(d) {
    const list = document.getElementById("pd-rank");
    const rows = byDataset[d].filter(p => p.e != null).sort((a, b) => a.e - b.e);
    const cap = Math.max(STATS[d].spread, 1e-9);
    const frag = document.createDocumentFragment();
    rows.forEach((p, i) => {
      const m = METHODS[p.m];
      const li = document.createElement("li");
      li.className = "pd-rankrow" + (p.m === state.contender ? " is-contender" : "");
      li.style.setProperty("--fam", FAM_VAR[m.family]);
      li.title = `${m.name} — ${fmtErr(p.e)} ${metricName(DATASETS[d])}, ${fmtNum(p.i, 1)}% behind the best` +
        (p.t != null ? `, ${fmtTime(p.t)} to fit` : "");
      const frac = Math.min(1, Math.max(0, 1 - (p.i || 0) / cap));
      li.innerHTML =
        `<span class="pos">${i + 1}</span>` +
        '<span class="who"><span class="dot"></span>' +
        '<span class="nm">' + (p.m === state.contender ? '<span class="star">&#9733;</span> ' : "") +
        `${escapeHtml(m.base)}</span>` +
        variantBadge(m) + "</span>" +
        `<span class="val">${fmtErr(p.e)}</span>` +
        `<span class="pd-rankbar"><span style="width:${(frac * 100).toFixed(1)}%"></span></span>`;
      li.dataset.t = String(trajIndexOf(p.m));
      li.addEventListener("click", () => togglePicked(trajIndexOf(p.m)));
      frag.appendChild(li);
    });
    list.appendChild(frag);
  }

  function trajIndexOf(methodIndex) {
    const found = trajIndexByName.get(METHODS[methodIndex].base);
    return found === undefined ? -1 : found;
  }

  function togglePicked(t) {
    if (t < 0) return;
    const d = state.selected;
    if (!state.picked.size) state.picked = new Set(activeTrajectories(d));
    if (state.picked.has(t)) state.picked.delete(t); else state.picked.add(t);
    drawChart();
  }

  // Which lines are drawn in colour: the Pareto front of the current y-axis plus the
  // contender, which stays visible however it did — finding out that it is nowhere near the
  // front is the whole point of looking at one dataset.
  function activeTrajectories(d) {
    const metric = Y_METRICS.find(m => m.key === state.yKey);
    const points = [];
    for (const [t, arr] of trajByDataset[d]) for (const pt of arr) points.push({ t: t, pt: pt });
    points.sort((a, b) => a.pt.x - b.pt.x || a.pt[metric.key] - b.pt[metric.key]);
    const front = new Set();
    let best = null;
    for (const { t, pt } of points) {
      const v = pt[metric.key];
      if (v == null) continue;
      if (best === null || v < best) { best = v; front.add(t); }
    }
    const contenderTraj = trajIndexOf(state.contender);
    if (contenderTraj >= 0) front.add(contenderTraj);
    return front;
  }

  function buildLegend(d) {
    const box = document.getElementById("legendstrip");
    if (!box) return;
    let html =
      '<span class="item"><svg width="26" height="9" viewBox="0 0 26 9">' +
      '<line x1="0" y1="4.5" x2="26" y2="4.5" stroke="var(--muted)" stroke-width="2"/>' +
      '<circle cx="6" cy="4.5" r="2.6" fill="var(--muted)"/><circle cx="17" cy="4.5" r="2.6" fill="var(--muted)"/>' +
      "</svg> Tuning trajectory (more configs &rarr; more time)</span>" +
      '<span class="item"><svg width="14" height="14" viewBox="0 0 14 14">' +
      `<path d="${starPath(7, 7, 6)}" fill="var(--contender)"/></svg> Contender</span>` +
      '<span class="item"><svg width="26" height="9" viewBox="0 0 26 9">' +
      '<line x1="0" y1="4.5" x2="26" y2="4.5" stroke="var(--ink)" stroke-width="1.6" stroke-dasharray="6 4"/>' +
      "</svg> Pareto front on this dataset</span>";
    const families = FAM_ORDER.filter(f => byDataset[d].some(p => METHODS[p.m].family === f));
    if (families.length > 1) {
      html += '<span class="legendbreak"></span><span class="item">Family:</span>';
      for (const fam of families) {
        html += `<span class="item"><svg width="12" height="12" viewBox="0 0 12 12">` +
          `<circle cx="6" cy="6" r="5" fill="${FAM_VAR[fam]}"/></svg> ` +
          `<span style="color:${FAM_INK[fam]}">${fam}</span></span>`;
      }
    }
    box.innerHTML = html;
  }

  // ---------- the chart ----------
  // The chart's own height. Fixed while the page is at its default size, and part of what "fit
  // to screen" trades against the list: on a laptop the detail pane alone is taller than the
  // viewport, so a fit that only shrank the list could only ever collapse it to nothing.
  const CHART_H_DEFAULT = 386, CHART_H_MIN = 220, LIST_FIT_MIN = 150;
  let chartHeight = CHART_H_DEFAULT;
  const M = { l: 66, r: 18, t: 14, b: 50 };

  function drawChart() {
    const d = state.selected;
    const svg = document.getElementById("pd-chart");
    const box = document.getElementById("pd-chartbox");
    if (!svg || d < 0) return;
    const perMethod = trajByDataset[d];
    svg.textContent = "";
    if (!perMethod || !perMethod.size) {
      box.innerHTML = '<p class="pd-empty">No tuning trajectories were published for this dataset.</p>';
      postHeight();
      return;
    }
    const metric = Y_METRICS.find(m => m.key === state.yKey);
    const rankList = document.getElementById("pd-rank");
    if (rankList) rankList.style.maxHeight = Math.max(160, chartHeight - 14) + "px";
    const textWidth = makeTextMeasurer(svg, { size: 12.5, weight: 700 });
    const active = state.picked.size ? state.picked : activeTrajectories(d);
    const all = [];
    for (const arr of perMethod.values()) for (const pt of arr) if (pt[metric.key] != null) all.push(pt);
    if (!all.length) { box.innerHTML = '<p class="pd-empty">Nothing to plot on this axis.</p>'; postHeight(); return; }

    const W = Math.max(340, Math.round(box.clientWidth || 620));
    const H = chartHeight;
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);

    const xs = all.map(p => p.x).filter(v => v > 0);
    const xmin = Math.min(...xs) * 0.65, xmax = Math.max(...xs) * 1.7;
    const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
    const X = v => M.l + (Math.log10(Math.max(v, xmin)) - lx0) / (lx1 - lx0) * (W - M.l - M.r);

    // The y-axis stops at the 90th percentile of the field rather than at its worst point. On a
    // single dataset one collapsed model is often an order of magnitude off, and letting it set
    // the range flattens every method worth comparing into a line along the bottom. Points
    // above the top are pinned there and drawn as hollow triangles, so they are visible as
    // "off the scale" instead of quietly dropped.
    const sortedVals = all.map(p => p[metric.key]).sort((a, b) => a - b);
    const lo = sortedVals[0], hi = sortedVals[sortedVals.length - 1];
    const ceiling = Math.max(quantile(sortedVals, 0.9), lo + (hi - lo) * 0.05, lo * 1.02);
    let y0, y1;
    if (metric.fromZero) {
      y0 = 0; y1 = ceiling * 1.07 || 1;
    } else {
      const pad = (ceiling - lo) * 0.08 || Math.abs(ceiling) * 0.08 || 1;
      y0 = lo - pad; y1 = ceiling + pad;
    }
    const rawY = v => M.t + (1 - (v - y0) / (y1 - y0)) * (H - M.t - M.b);
    const Y = v => Math.min(H - M.b, Math.max(M.t, rawY(v)));
    const offScale = v => v > y1;
    let clipped = 0;

    const grid = el("g", {}, svg);
    for (let e = Math.ceil(lx0); Math.pow(10, e) < xmax; e++) {
      const gx = X(Math.pow(10, e));
      el("line", { x1: gx, y1: M.t, x2: gx, y2: H - M.b, stroke: "var(--line)", "stroke-width": 1 }, grid);
      el("text", {
        x: gx, y: H - M.b + 19, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)",
      }, grid).textContent = fmtNum(Math.pow(10, e), e >= 0 ? 0 : -e);
    }
    for (const yv of ticks(y0, y1, 6)) {
      const gy = Y(yv);
      el("line", { x1: M.l, y1: gy, x2: W - M.r, y2: gy, stroke: "var(--line)", "stroke-width": 1 }, grid);
      el("text", {
        x: M.l - 8, y: gy + 4, "text-anchor": "end", "font-size": 12, fill: "var(--muted)",
      }, grid).textContent = metric.key === "e" ? fmtErr(yv) : fmtNum(yv, Number.isInteger(yv) ? 0 : 1);
    }
    el("rect", {
      x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "none", stroke: "var(--line)",
    }, grid);
    el("text", {
      x: (M.l + W - M.r) / 2, y: H - 9, "text-anchor": "middle", "font-size": 13.5,
      "font-weight": 650, fill: "var(--ink)",
    }, grid).textContent = "Median training time on this dataset (s) — lower is better";
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": 13.5, "font-weight": 650, fill: "var(--ink)",
      transform: `translate(16 ${(M.t + H - M.b) / 2}) rotate(-90)`,
    }, grid).textContent =
      (metric.key === "e" ? `Test error (${metricName(DATASETS[d])})` : "Improvability (%)") + " — lower is better";

    // The dataset's own Pareto front, drawn like the aggregate figure's.
    const ordered = all.slice().sort((a, b) => a.x - b.x || a[metric.key] - b[metric.key]);
    const verts = [];
    let running = null;
    for (const pt of ordered) {
      const v = pt[metric.key];
      if (running === null || v < running) {
        if (running !== null) verts.push([pt.x, running]);
        verts.push([pt.x, v]);
        running = v;
      }
    }
    if (verts.length) {
      let path = `M${X(verts[0][0])},${M.t}`;
      for (const [vx, vy] of verts) path += ` L${X(vx)},${Y(vy)}`;
      path += ` L${W - M.r},${Y(verts[verts.length - 1][1])}`;
      el("path", {
        d: path, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6, "stroke-dasharray": "7 5", opacity: 0.8,
      }, svg);
    }

    const contenderTraj = trajIndexOf(state.contender);
    const lines = el("g", {}, svg);
    const marksOff = el("g", {}, svg);
    const marksOn = el("g", {}, svg);
    for (const [t, arr] of perMethod) {
      const pts = arr.filter(p => p[metric.key] != null);
      if (!pts.length) continue;
      const on = active.has(t);
      const fam = famOfTrajectory(t);
      const color = on ? FAM_VAR[fam] : "var(--pt-muted)";
      if (pts.length > 1) {
        el("path", {
          d: pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(p[metric.key])}`).join(" "),
          fill: "none", stroke: color, "stroke-width": on ? 2 : 1, opacity: on ? 0.65 : 0.3, "data-t": t,
        }, lines);
      }
      for (const p of pts) {
        const cx = X(p.x), cy = Y(p[metric.key]);
        if (offScale(p[metric.key])) {
          clipped++;
          const s = on ? 5 : 4;
          el("path", {
            d: `M${cx},${cy - s} L${cx + s},${cy + s * 0.7} L${cx - s},${cy + s * 0.7} Z`,
            fill: "none", stroke: color, "stroke-width": 1.4, opacity: on ? 0.9 : 0.45, "data-t": t,
          }, on ? marksOn : marksOff);
          continue;
        }
        el("circle", {
          cx: cx, cy: cy, r: on ? 4.2 : 3, fill: color, opacity: on ? 0.95 : 0.45, "data-t": t,
        }, on ? marksOn : marksOff);
      }
      if (t === contenderTraj) {
        const bestPt = pts.reduce((a, b) => (b[metric.key] < a[metric.key] ? b : a));
        el("path", {
          d: starPath(X(bestPt.x), Y(bestPt[metric.key]), 8.5), fill: "var(--contender)",
          stroke: "var(--card)", "stroke-width": 1.2, "data-t": t,
        }, marksOn);
      }
    }

    // Direct labels for the coloured lines, measured so a long system name is put on whichever
    // side of its point has room and never runs off the plot.
    const labels = [];
    for (const t of active) {
      const arr = (perMethod.get(t) || []).filter(p => p[metric.key] != null);
      if (!arr.length) continue;
      const bestPt = arr.reduce((a, b) => (b[metric.key] < a[metric.key] ? b : a));
      const name = TRAJ_METHODS[t];
      const w = textWidth(name);
      const px = X(bestPt.x);
      const toRight = px + 10 + w <= W - M.r;
      labels.push({
        name: name, family: famOfTrajectory(t), w: w,
        anchor: toRight ? "start" : "end",
        x: toRight ? px + 10 : Math.max(M.l + w + 2, px - 10),
        y: Y(bestPt[metric.key]) - 11,
      });
    }
    const spanOf = l => (l.anchor === "start" ? [l.x, l.x + l.w] : [l.x - l.w, l.x]);
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      for (let j = 0; j < i; j++) {
        const [aL, aR] = spanOf(labels[i]), [bL, bR] = spanOf(labels[j]);
        if (aL < bR + 6 && bL < aR + 6 && Math.abs(labels[i].y - labels[j].y) < 14) {
          labels[i].y = labels[j].y + 14;
        }
      }
    }
    const labelG = el("g", {}, svg);
    for (const l of labels) {
      el("text", {
        x: l.x, y: Math.max(l.y, M.t + 11), "font-size": 12.5, "font-weight": 700, fill: FAM_VAR[l.family],
        "paint-order": "stroke", stroke: "var(--card)", "stroke-width": 3.5, "text-anchor": l.anchor,
      }, labelG).textContent = l.name;
    }

    const hits = el("g", {}, svg);
    for (const [t, arr] of perMethod) {
      for (const p of arr) {
        if (p[metric.key] == null) continue;
        const hit = el("circle", {
          cx: X(p.x), cy: Y(p[metric.key]), r: 11, fill: "transparent", cursor: "pointer",
        }, hits);
        hit.addEventListener("mouseenter", ev => tip.show(chartTip(d, t, p), ev));
        hit.addEventListener("mousemove", ev => tip.move(ev));
        hit.addEventListener("mouseleave", () => tip.hide());
        hit.addEventListener("click", () => togglePicked(t));
      }
    }
    markOffScale(clipped);
    syncRankHighlight(active);
    postHeight();
  }

  // The legend gains an "off the scale" entry only while something is actually pinned to the
  // top edge, and loses it again when the axis changes and nothing is.
  function markOffScale(count) {
    const legend = document.getElementById("legendstrip");
    if (!legend) return;
    const existing = legend.querySelector(".pd-offscale");
    if (!count) { if (existing) existing.remove(); return; }
    if (existing) return;
    legend.insertAdjacentHTML("beforeend",
      '<span class="item pd-offscale"><svg width="14" height="14" viewBox="0 0 14 14">' +
      '<path d="M7,3 L12,10 L2,10 Z" fill="none" stroke="var(--muted)" stroke-width="1.4"/></svg>' +
      " Off the top of the scale (hover for the value)</span>");
  }

  // A trajectory line's family: taken from a leaderboard method that shares its base name, so
  // the line, the ranking list and the strip agree on the colour.
  const famCache = new Map();
  function famOfTrajectory(t) {
    const name = TRAJ_METHODS[t];
    if (famCache.has(name)) return famCache.get(name);
    const match = METHODS.find(m => m.base === name);
    const fam = match ? match.family : FAM_MERGED;
    famCache.set(name, fam);
    return fam;
  }

  function chartTip(d, t, p) {
    return `<div class="t-name">${escapeHtml(TRAJ_METHODS[t])}` +
      (p.n != null ? ` <span class="t-var">(${p.n} config${p.n === 1 ? "" : "s"})</span>` : "") + "</div>" +
      `<div>Test error: <b>${fmtErr(p.e)}</b></div>` +
      `<div>Improvability: <b>${fmtNum(p.i, 1)}%</b></div>` +
      `<div>Training time: <b>${fmtTime(p.x)}</b></div>`;
  }

  function syncRankHighlight(active) {
    for (const li of document.querySelectorAll("#pd-rank .pd-rankrow")) {
      li.classList.toggle("is-active", active.has(Number(li.dataset.t)));
    }
  }

  // ---------- controls ----------
  function buildContenderSelect() {
    const select = document.getElementById("contender");
    const order = METHODS.map((m, i) => i).sort((a, b) => {
      const fa = FAM_ORDER.indexOf(METHODS[a].family), fb = FAM_ORDER.indexOf(METHODS[b].family);
      return fa - fb || METHODS[a].name.localeCompare(METHODS[b].name);
    });
    let currentFamily = null, group = null;
    for (const i of order) {
      const m = METHODS[i];
      if (m.family !== currentFamily) {
        currentFamily = m.family;
        group = document.createElement("optgroup");
        group.label = currentFamily;
        select.appendChild(group);
      }
      const opt = document.createElement("option");
      opt.value = String(i);
      opt.textContent = m.name;
      group.appendChild(opt);
    }
    select.value = String(state.contender);
    select.addEventListener("change", ev => {
      state.contender = Number(ev.target.value);
      // A held selection of lines belonged to the previous contender; drop it so the chart
      // goes back to "the front plus whoever is the contender now".
      state.picked = new Set();
      renderList();
      renderDetail();
    });
  }

  function buildFilter(elementId, label, options, field) {
    const box = document.getElementById(elementId);
    if (options.length < 2) { box.hidden = true; return; }
    box.innerHTML = `<span class="grouplabel">${label}</span>`;
    for (const opt of options) {
      const button = document.createElement("button");
      button.className = "pd-fchip";
      button.type = "button";
      button.textContent = opt.label;
      button.dataset.v = opt.key;
      button.addEventListener("click", () => {
        state[field] = opt.key;
        syncFilter(box, field);
        renderList();
      });
      box.appendChild(button);
    }
    syncFilter(box, field);
  }
  function syncFilter(box, field) {
    for (const button of box.querySelectorAll(".pd-fchip")) {
      button.setAttribute("aria-pressed", String(button.dataset.v === state[field]));
    }
  }

  const TASK_LABELS = { binary: "Binary", multiclass: "Multiclass", regression: "Regression" };
  const tasksPresent = [...new Set(DATASETS.map(ds => ds.task).filter(Boolean))]
    .sort((a, b) => (TASK_LABELS[a] || a).localeCompare(TASK_LABELS[b] || b));
  buildFilter("taskfilter", "Task", [
    { key: "all", label: "All" },
    ...tasksPresent.map(t => ({ key: t, label: TASK_LABELS[t] || t })),
  ], "task");
  const sizesPresent = SIZE_BUCKETS.filter(b => SIZE_OF.includes(b.key));
  buildFilter("sizefilter", "Size", [
    { key: "all", label: "All" },
    ...sizesPresent.map(b => ({ key: b.key, label: b.label })),
  ], "size");

  let searchTimer = null;
  document.getElementById("search").addEventListener("input", ev => {
    const value = ev.target.value.trim().toLowerCase();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.query = value; renderList(); }, 110);
  });

  // The host page can preselect the filters so the browser opens on the same slice of the
  // benchmark the reader already chose in the leaderboard's task and dataset tabs.
  window.addEventListener("message", ev => {
    const data = ev.data;
    if (data && data.type === "tabarena-perdataset-viewport" && typeof data.height === "number") {
      hostViewport = data.height;
      if (fitBtn.getAttribute("aria-pressed") === "true") fitToScreen();
      return;
    }
    if (!data || data.type !== "tabarena-perdataset-filter") return;
    // The host re-sends this every time the frame reports a height, so ignore the ones that
    // ask for what is already on screen rather than re-rendering the list for nothing.
    if (data.task === state.task && data.size === state.size) return;
    if (typeof data.task === "string") state.task = data.task;
    if (typeof data.size === "string") state.size = data.size;
    syncFilter(document.getElementById("taskfilter"), "task");
    syncFilter(document.getElementById("sizefilter"), "size");
    renderList();
  });

  // ---------- keyboard ----------
  // The click is spelled out as a precondition, not just as an alternative way to select. This
  // page is a sandboxed frame inside the site, so it receives no key events at all until the
  // reader has clicked inside it — without that sentence the arrow keys look broken.
  document.getElementById("listhint").innerHTML =
    "Click a dataset to see how the field got there · after that first click, " +
    "<kbd>&larr;</kbd> <kbd>&rarr;</kbd> step through the list and " +
    "<kbd>Home</kbd> <kbd>End</kbd> jump to the ends · sort by clicking a column · " +
    "the list scrolls, and the bar below it resizes it";

  document.addEventListener("keydown", ev => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
      if (ev.key === "Escape") ev.target.blur();
      return;
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const moves = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, PageDown: 10, PageUp: -10 };
    if (ev.key in moves) { ev.preventDefault(); step(moves[ev.key]); return; }
    if (ev.key === "Home") { ev.preventDefault(); if (visible.length) select(visible[0]); return; }
    if (ev.key === "End") { ev.preventDefault(); if (visible.length) select(visible[visible.length - 1]); }
  });

  // ---------- chrome ----------
  // Opens with the controls: browsing is the point of this page, not a single figure.
  setUpPaperView(() => drawChart(), { openInPaper: false });
  setUpExport(() => {
    const svg = document.getElementById("pd-chart");
    return svg ? [{ svg: svg, dx: 0 }] : [];
  }, () => slugify((DATASETS[state.selected] || {}).name || document.title));

  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(drawChart, 120);
  });
  window.addEventListener("load", () => { sizeList(); postHeight(); });

  buildContenderSelect();
  buildHead();
  renderList();
  // The frame is often still being laid out when the first render runs, so the list is sized
  // again on the next frame, once its rows have a height.
  requestAnimationFrame(sizeList);
})();
</script>
</body>
</html>
"""
