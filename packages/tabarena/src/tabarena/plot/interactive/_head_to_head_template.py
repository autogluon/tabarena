"""HTML/CSS/JS template for the self-contained head-to-head browser.

Two methods picked by the reader, A and B, compared dataset by dataset. From the top: a summary
over the datasets in view (the paired Elo gap, the error ratio, the datasets each leads and leads
significantly, the split win rate, time and rank); a chart of a per-dataset quantity (margin, rank
or improvability difference, pull on the Elo gap, speed-up) against the datasets sorted, their size
or their metadata; the Elo gap for every subset of the datasets, whose rows keep or take out their
datasets; A against every other method; a sortable list of the datasets; and the selected dataset's
splits and metadata. The view is kept in the page address.

Kept as a Python string constant (rather than a package-data file) so it ships with the package
without any build-system data-file configuration.
:func:`tabarena.plot.interactive.head_to_head_explorer.build_head_to_head_explorer_html`
substitutes the placeholders (see
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

HEAD_TO_HEAD_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  /* The two sides. Fixed colours whoever A and B are: the family colours the rest of the site
     uses would make two foundation models the same purple. Blue and orange, the first two slots
     of the categorical palette, kept apart from the green and red a page reads as good and bad.
     Declared across the same four scopes as the shared tokens (see _explorer_shared). */
  :root { --side-a: #2a78d6; --side-b: #eb6834; --exclude: #d6363c; }
  @media (prefers-color-scheme: dark) {
    :root:not([data-theme="light"]) { --side-a: #3987e5; --side-b: #d95926; --exclude: #f0646a; }
  }
  :root[data-theme="dark"] { --side-a: #3987e5; --side-b: #d95926; --exclude: #f0646a; }
  :root[data-theme="light"] { --side-a: #2a78d6; --side-b: #eb6834; --exclude: #d6363c; }

  body { padding: 8px 8px 12px; }
  .hh-page { position: relative; }

  .hh-controls { display: flex; flex-direction: column; gap: 8px; margin-bottom: 8px; }
  .hh-pickers { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 10px; }
  .hh-side {
    display: inline-flex; align-items: center; justify-content: center; width: 20px; height: 20px; flex: none;
    border-radius: 50%; font: 700 11.5px/1 system-ui, sans-serif; color: #fff;
  }
  .hh-side.a { background: var(--side-a); }
  .hh-side.b { background: var(--side-b); }
  .hh-pick { display: inline-flex; align-items: center; gap: 6px; min-width: 0; }
  .hh-pick select { max-width: min(360px, 70vw); }
  .hh-vs { font-size: 12.5px; font-weight: 650; color: var(--muted); }
  .hh-filters { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 16px; }
  .hh-filtergroup { display: inline-flex; flex-wrap: wrap; align-items: center; gap: 4px; }
  .hh-filtergroup .grouplabel { font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; margin-right: 2px; }
  .hh-fchip {
    font: 600 11.5px/1 system-ui, sans-serif; color: var(--muted); background: none;
    border: 1px solid var(--line); border-radius: 999px; padding: 4px 9px; cursor: pointer;
  }
  .hh-fchip:hover { border-color: var(--muted); color: var(--ink); }
  .hh-fchip[aria-pressed="true"] {
    border-color: var(--accent); color: var(--ink); background: color-mix(in srgb, var(--accent) 15%, transparent);
  }
  .hh-fchip .n { margin-left: 4px; font-weight: 500; opacity: 0.7; font-variant-numeric: tabular-nums; }
  .hh-fchip[aria-disabled="true"] { opacity: 0.4; cursor: default; }
  .hh-search {
    font: 500 12.5px/1.2 system-ui, sans-serif; color: var(--ink); background: var(--chip-bg);
    border: 1px solid var(--line); border-radius: 7px; padding: 6px 9px; min-width: 180px; flex: 1 1 180px; max-width: 300px;
  }
  .hh-search:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .hh-check { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; color: var(--muted); cursor: pointer; }
  .hh-check input { accent-color: var(--accent); }

  /* --- Summary ------------------------------------------------------------------ */
  .hh-summary {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(170px, 1fr)); gap: 8px; margin: 4px 0 10px;
  }
  .hh-tile { border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 8px 12px 9px; min-width: 0; }
  .hh-tlabel {
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  }
  .hh-tvalue { font-size: 19px; font-weight: 700; font-variant-numeric: tabular-nums; line-height: 1.3; }
  .hh-tsub { font-size: 11.5px; color: var(--muted); font-variant-numeric: tabular-nums; }
  .hh-swatch { display: inline-block; width: 9px; height: 9px; border-radius: 2px; margin-right: 4px; vertical-align: 0; }
  .hh-swatch.a { background: var(--side-a); }
  .hh-swatch.b { background: var(--side-b); }

  /* --- Chart ---------------------------------------------------------------------- */
  .hh-chartcard { border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 8px 12px 10px; }
  .hh-chartbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin-bottom: 4px; }
  .hh-chartbar .hint { font-size: 12px; color: var(--muted); }
  .chartbox { position: relative; }
  .chartbox svg { display: block; }

  /* --- Table + detail --------------------------------------------------------------- */
  .hh-main { margin-top: 12px; display: flex; flex-direction: column; gap: 12px; }
  .hh-listwrap {
    overflow: auto; max-height: 460px; border: 1px solid var(--line); border-radius: 10px;
    scrollbar-color: var(--muted) transparent;
  }
  .hh-listwrap:focus-visible { outline: 2px solid var(--accent); outline-offset: 1px; }
  table.hh-table { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  table.hh-table thead th {
    position: sticky; top: 0; z-index: 3; background: var(--card); color: var(--muted);
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.05em; text-transform: uppercase;
    text-align: left; padding: 7px 5px; white-space: nowrap; cursor: pointer; user-select: none;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  table.hh-table thead th:hover { color: var(--accent); }
  table.hh-table thead th.sortable::after { content: "\2195"; font-size: 0.8em; opacity: 0.35; margin-left: 3px; }
  table.hh-table thead th[aria-sort="ascending"]::after { content: "\25B2"; opacity: 0.9; }
  table.hh-table thead th[aria-sort="descending"]::after { content: "\25BC"; opacity: 0.9; }
  table.hh-table td { padding: 4px 5px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  table.hh-table tbody tr { cursor: pointer; scroll-margin-top: 30px; }
  table.hh-table tbody tr:hover td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
  table.hh-table tbody tr.sel td { background: color-mix(in srgb, var(--accent) 17%, transparent); }
  table.hh-table tbody tr.sel td:first-child { box-shadow: inset 3px 0 0 var(--accent); }
  td.hh-num, th.hh-num { text-align: right; }
  .hh-pos { color: var(--muted); }
  .hh-name { font-weight: 620; display: inline-block; max-width: 200px; overflow: hidden; text-overflow: ellipsis; vertical-align: bottom; }
  .hh-tag {
    display: inline-block; font-size: 10.5px; font-weight: 650; border: 1px solid var(--line);
    border-radius: 5px; padding: 1px 5px; color: var(--muted);
  }
  /* A value in a side's favour carries that side's mark, and a significant one says so in weight. */
  .hh-lead { display: inline-flex; align-items: center; gap: 5px; justify-content: flex-end; }
  .hh-lead::before { content: ""; width: 7px; height: 7px; border-radius: 2px; background: var(--lead, transparent); }
  .hh-lead.a { --lead: var(--side-a); }
  .hh-lead.b { --lead: var(--side-b); }
  .hh-sig { font-weight: 700; color: var(--ink); }
  .hh-nosig { color: var(--muted); }
  .hh-verdict { font-size: 11.5px; font-weight: 650; border-radius: 999px; padding: 1px 8px; white-space: nowrap; }
  .hh-verdict.a { background: color-mix(in srgb, var(--side-a) 18%, transparent); color: var(--ink); }
  .hh-verdict.b { background: color-mix(in srgb, var(--side-b) 20%, transparent); color: var(--ink); }
  .hh-verdict.tie { color: var(--muted); border: 1px solid var(--line); }
  .hh-verdict.na { color: var(--muted); }
  .hh-count { color: var(--muted); font-size: 12px; }

  .hh-detail { border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 9px 14px 12px; }
  .hh-dhead { display: flex; flex-wrap: wrap; align-items: baseline; gap: 2px 10px; margin-bottom: 6px; }
  .hh-dname { font-size: 15px; font-weight: 700; }
  .hh-dsub { font-size: 12px; color: var(--muted); }
  .hh-facts { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 0 22px; margin: 0 0 8px; }
  .hh-fact {
    display: flex; justify-content: space-between; gap: 10px; padding: 2px 0; font-size: 12px;
    border-bottom: 1px dashed color-mix(in srgb, var(--line) 80%, transparent);
  }
  .hh-fact dt { color: var(--muted); white-space: nowrap; }
  .hh-fact dd { margin: 0; font-weight: 620; font-variant-numeric: tabular-nums; text-align: right; white-space: nowrap; }
  .hh-empty { color: var(--muted); font-size: 12.5px; font-style: italic; padding: 18px 4px; }
  .hh-listfoot { display: flex; flex-wrap: wrap; justify-content: space-between; align-items: center; gap: 6px 12px; margin-top: 6px; }
  table.hh-subtable th.sortable { cursor: pointer; }
  table.hh-subtable th.sortable:hover { color: var(--accent); }
  table.hh-subtable th[aria-sort="ascending"]::after { content: " \25B2"; }
  table.hh-subtable th[aria-sort="descending"]::after { content: " \25BC"; }
  table.hh-subtable tr.is-b td { background: color-mix(in srgb, var(--side-b) 14%, transparent); }
  .hh-splitbar { display: flex; flex-wrap: wrap; align-items: center; gap: 6px 14px; margin: 2px 0 4px; }
  /* The dataset's task metadata: column counts, missingness, target and split facts. */
  .hh-meta { margin-top: 8px; }
  .hh-meta summary {
    cursor: pointer; font-size: 10.5px; font-weight: 650; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted);
  }
  .hh-metagrid { display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 0 22px; margin: 4px 0 0; }
  .hh-metagrid dt { min-width: 0; overflow: hidden; text-overflow: ellipsis; }
  .hh-metagrid dd { flex: none; max-width: 60%; overflow: hidden; text-overflow: ellipsis; }

  @media (min-width: 1100px) {
    .hh-main { display: grid; grid-template-columns: minmax(0, 64fr) minmax(0, 36fr); align-items: start; }
    .hh-listwrap { max-height: 620px; }
  }
  /* The list beside the detail has room for every column only on a wide screen: the secondary columns
     (task, features, both ranks, the time ratio) go first, then the split counts and the p-value. */
  @media (max-width: 1999px) {
    .hh-wide { display: none; }
  }
  @media (max-width: 1399px) {
    .hh-mid { display: none; }
  }
  #better-fact { white-space: normal; }
  /* --- Elo gap by subset: one forest-plot row per way of keeping the datasets -------------- */
  .hh-subsets { margin-top: 12px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); padding: 8px 12px 10px; }
  .hh-subsets summary { cursor: pointer; font-size: 13px; font-weight: 650; }
  .hh-subsets summary .hint { font-size: 12px; font-weight: 400; color: var(--muted); margin-left: 8px; }
  .hh-subwrap { overflow-x: auto; margin-top: 6px; }
  table.hh-subtable { border-collapse: collapse; width: 100%; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  table.hh-subtable th {
    text-align: left; font-size: 10.5px; font-weight: 650; letter-spacing: 0.05em; text-transform: uppercase;
    color: var(--muted); padding: 4px 8px; border-bottom: 1px solid var(--line); white-space: nowrap;
  }
  table.hh-subtable td { padding: 3px 8px; border-bottom: 1px solid var(--line); white-space: nowrap; }
  table.hh-subtable tr.group td { padding-top: 8px; font-size: 10.5px; font-weight: 650; letter-spacing: 0.06em;
    text-transform: uppercase; color: var(--muted); border-bottom: 0; }
  table.hh-subtable tr.clickable { cursor: pointer; }
  table.hh-subtable tr.clickable:hover td { background: color-mix(in srgb, var(--accent) 8%, transparent); }
  table.hh-subtable tr.clickable:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  /* A kept subset (first click) is tinted with the accent; a subtracted one (second click) in red. */
  table.hh-subtable tr.mark-in td { background: color-mix(in srgb, var(--accent) 14%, transparent); }
  table.hh-subtable tr.mark-in td:first-child { box-shadow: inset 3px 0 0 var(--accent); font-weight: 650; }
  table.hh-subtable tr.mark-out td { background: color-mix(in srgb, var(--exclude) 16%, transparent); }
  table.hh-subtable tr.mark-out td:first-child { box-shadow: inset 3px 0 0 var(--exclude); font-weight: 650; }
  .hh-markbadge { display: inline-block; width: 15px; font-weight: 700; }
  tr.mark-in .hh-markbadge { color: var(--accent); }
  tr.mark-out .hh-markbadge { color: var(--exclude); }
  /* The marked subsets, as removable chips beside the filters. */
  .hh-marks { display: flex; flex-wrap: wrap; align-items: center; gap: 4px 6px; }
  .hh-marks .grouplabel { font-size: 10.5px; letter-spacing: 0.06em; text-transform: uppercase; margin-right: 2px; }
  .hh-mark {
    font: 600 11.5px/1 system-ui, sans-serif; color: var(--ink); border-radius: 999px; padding: 4px 9px; cursor: pointer;
    border: 1px solid var(--accent); background: color-mix(in srgb, var(--accent) 14%, transparent);
  }
  .hh-mark.out { border-color: var(--exclude); background: color-mix(in srgb, var(--exclude) 16%, transparent); }
  .hh-mark:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  table.hh-subtable td.num, table.hh-subtable th.num { text-align: right; }
  table.hh-subtable td.excludes0 { font-weight: 700; }
  table.hh-subtable td.forest { width: 240px; padding-top: 0; padding-bottom: 0; }
  table.hh-subtable .pending { color: var(--muted); }
  body.paper .hh-controls, body.paper .hh-main, body.paper .hh-chartbar, body.paper .hh-subsets { display: none !important; }
  body.paper .hh-chartcard { border: none; background: none; padding: 0; }
</style>
</head>
<body>
<div class="hh-page" id="page">
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, the summary chart only">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>

  <div class="hh-controls">
    <div class="hh-pickers">
      <label class="metricpick hh-pick"><span class="hh-side a">A</span><select id="pick-a" aria-label="Method A"></select></label>
      <button class="btn" id="btn-swap" title="Swap A and B">&#8646; Swap</button>
      <label class="metricpick hh-pick"><span class="hh-side b">B</span><select id="pick-b" aria-label="Method B"></select></label>
    </div>
    <div class="hh-filters">
      <input class="hh-search" id="search" type="search" placeholder="Find by name, domain or source" aria-label="Find a dataset">
      <div class="hh-filtergroup" id="taskfilter"></div>
      <div class="hh-filtergroup" id="splitfilter"></div>
      <div class="hh-filtergroup" id="sizefilter"></div>
      <div class="hh-filtergroup" id="balancefilter"></div>
      <div class="hh-filtergroup" id="kindfilter"></div>
      <label class="hh-check" title="Call a dataset significant only when its Benjamini-Hochberg adjusted p-value, over the datasets in view, is below the level">
        <input type="checkbox" id="bh"> Adjust for the number of datasets (BH)</label>
    </div>
    <div class="hh-marks" id="marks" hidden></div>
  </div>

  <div class="hh-summary" id="summary"></div>

  <div class="hh-chartcard">
    <div class="hh-chartbar">
      <label class="metricpick">Y-axis <select id="yaxis"></select></label>
      <label class="metricpick">X-axis <select id="xaxis"></select></label>
      <span class="hint">Faded: the splits cannot tell the two apart. Click a dataset to see its splits.</span>
    </div>
    <div class="legendstrip" id="legendstrip"></div>
    <div class="exportbar" id="exportbar" hidden>
      <span class="hint">Export the chart</span>
      <button class="btn" id="btn-svg">SVG</button>
      <button class="btn" id="btn-pdf">PDF</button>
      <button class="btn" id="btn-png">PNG</button>
    </div>
    <div class="chartbox" id="chartbox"><svg id="chart" role="img" aria-label="Margin per dataset"></svg></div>
  </div>

  <details class="hh-subsets" id="subsets" open>
    <summary>Elo gap A &minus; B by subset<span class="hint">paired, each against the field without the other; bold:
      the interval excludes 0; click a row once to keep only those datasets, twice to take them out (red), a third time
      to clear; marks combine</span></summary>
    <div class="hh-subwrap"><table class="hh-subtable"><thead><tr>
      <th>Datasets kept</th><th class="num">Datasets</th><th class="num">Elo gap</th><th class="num">95% interval</th>
      <th id="forest-head"></th><th class="num">A ahead in</th><th class="num">Error ratio A / B</th>
    </tr></thead><tbody id="subset-body"></tbody></table></div>
  </details>

  <details class="hh-subsets" id="vsfield">
    <summary>A against every method<span class="hint">over the datasets in view, computed when opened (Elo intervals from
      200 resamples); sort by a column, for example the methods significantly better than A; click a row to make it B</span></summary>
    <div class="hh-subwrap"><table class="hh-subtable"><thead id="vs-head"></thead><tbody id="vs-body"></tbody></table></div>
  </details>

  <div class="hh-main">
    <div>
      <div class="hh-listwrap" id="listwrap" tabindex="0" aria-label="Datasets">
        <table class="hh-table"><thead id="thead"></thead><tbody id="tbody"></tbody></table>
      </div>
      <div class="hh-listfoot">
        <div class="hh-count" id="count"></div>
        <div class="btnrow">
          <button class="btn" id="btn-copy-csv" type="button" title="Copy the list as it is sorted and filtered, as CSV">Copy table</button>
          <button class="btn" id="btn-csv" type="button" title="Save the list as it is sorted and filtered, as a CSV file">Download CSV</button>
        </div>
      </div>
    </div>
    <div class="hh-detail" id="detail"></div>
  </div>
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
  for (const m of METHODS) if (leakageOf(m.base)) m.name = m.base + leakageMark(m.base) + m.name.slice(m.base.length);
  for (const m of METHODS) m.family = famOf(m.family);
  const METRIC_DISPLAY = CONFIG.metricDisplay || {};
  const SIZE_BUCKETS = CONFIG.sizeBuckets || [];
  const SIG = CONFIG.significance || {};
  const ALPHA = SIG.alpha != null ? SIG.alpha : 0.05;
  const MIN_SPLITS = SIG.minSplits || 6;
  const EXACT_MAX = SIG.exactMax || 50;
  const RESAMPLES = CONFIG.bootstrap || 2000;
  const SPLIT_LABELS = { random: "IID", temporal: "Temporal", grouped: "Grouped" };
  const CLASSIFICATION_TASKS = new Set(["binary", "multiclass"]);
  const TASK_LABELS = { binary: "Binary", multiclass: "Multiclass", regression: "Regression" };
  const TASK_SHORT = { binary: "binary", multiclass: "multi", regression: "reg" };
  const nD = DATASETS.length;

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;
  const page = document.getElementById("page");
  const tip = makeTooltip(page);

  const cell = new Map();
  for (const p of POINTS) cell.set(p.d + ":" + p.m, p);
  const key = (d, m) => cell.get(d + ":" + m);

  // ---------- Elo ----------
  // The leaderboard's Elo (bencheval's rank path): on a split a method wins against every method
  // with a higher error and half against every tie; a dataset's win totals are the means over its
  // splits; a Bradley-Terry fit with bencheval's small ridge turns the summed totals into ratings,
  // shifted so that the calibration method sits at 1000. The field is every method with a result
  // on every split, imputed results included, as on the leaderboard. A field member's split errors
  // are its rows of this grid.
  const ELO = CONFIG.elo || null;
  const ELO_POS = new Map((ELO ? ELO.field : []).map((m, k) => [m, k]));
  if (ELO) for (const p of POINTS) if (!p.f && ELO_POS.has(p.m)) p.f = ELO.grid[p.d][ELO_POS.get(p.m)];
  const BT_RIDGE = 0.5 / (1e6 * Math.LN10 * Math.LN10);
  // Each dataset's win totals over the whole field.
  const ELO_WINS = ELO ? ELO.grid.map(rows => {
    const n = rows.length, splits = n ? rows[0].length : 0;
    const wins = new Float64Array(n);
    for (let s = 0; s < splits; s++) {
      for (let i = 0; i < n; i++) {
        const ei = rows[i][s];
        for (let j = i + 1; j < n; j++) {
          const ej = rows[j][s];
          if (ei < ej) wins[i] += 1; else if (ej < ei) wins[j] += 1; else { wins[i] += 0.5; wins[j] += 0.5; }
        }
      }
    }
    for (let i = 0; i < n; i++) wins[i] /= splits || 1;
    return wins;
  }) : [];
  // Each field member's share of wins against field member `x` on dataset `d`: what leaving `x`
  // out of the field takes off its win totals.
  function winsAgainst(d, x) {
    const rows = ELO.grid[d], n = rows.length, splits = rows[x].length;
    const out = new Float64Array(n);
    for (let s = 0; s < splits; s++) {
      const ex = rows[x][s];
      for (let k = 0; k < n; k++) {
        if (k === x) continue;
        const ek = rows[k][s];
        if (ek < ex) out[k] += 1; else if (ek === ex) out[k] += 0.5;
      }
    }
    for (let k = 0; k < n; k++) out[k] /= splits || 1;
    return out;
  }
  // Ridge-penalised Bradley-Terry log-strengths from win totals, every pair meeting `nTasks` times
  // (bencheval's `_bradley_terry_from_win_totals`), by Newton's method.
  function bradleyTerry(wins, nTasks, init) {
    const n = wins.length;
    const t = init ? Float64Array.from(init) : new Float64Array(n);
    const g = new Float64Array(n), H = new Float64Array(n * n);
    for (let iter = 0; iter < 200; iter++) {
      g.fill(0);
      H.fill(0);
      for (let i = 0; i < n; i++) {
        for (let j = i + 1; j < n; j++) {
          const p = 1 / (1 + Math.exp(t[j] - t[i]));
          const w = nTasks * p * (1 - p);
          g[i] += nTasks * p;
          g[j] += nTasks * (1 - p);
          H[i * n + i] += w; H[j * n + j] += w; H[i * n + j] -= w; H[j * n + i] -= w;
        }
      }
      let largest = 0;
      for (let i = 0; i < n; i++) {
        g[i] += 2 * BT_RIDGE * t[i] - wins[i];
        H[i * n + i] += 2 * BT_RIDGE;
        largest = Math.max(largest, Math.abs(g[i]));
      }
      if (largest < 1e-10 * Math.max(1, nTasks)) break;
      const step = choleskySolve(H, g, n);
      for (let i = 0; i < n; i++) t[i] -= step[i];
    }
    return t;
  }
  function choleskySolve(A, b, n) {
    const L = new Float64Array(n * n);
    for (let i = 0; i < n; i++) {
      for (let j = 0; j <= i; j++) {
        let sum = A[i * n + j];
        for (let k = 0; k < j; k++) sum -= L[i * n + k] * L[j * n + k];
        L[i * n + j] = i === j ? Math.sqrt(Math.max(sum, 1e-300)) : sum / L[j * n + j];
      }
    }
    const y = new Float64Array(n), x = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      let sum = b[i];
      for (let k = 0; k < i; k++) sum -= L[i * n + k] * y[k];
      y[i] = sum / L[i * n + i];
    }
    for (let i = n - 1; i >= 0; i--) {
      let sum = y[i];
      for (let k = i + 1; k < n; k++) sum -= L[k * n + i] * x[k];
      x[i] = sum / L[i * n + i];
    }
    return x;
  }
  // Ratings of a board (field positions `members`) from its log-strengths: centred on 1000, then
  // shifted so the calibration method sits at 1000 when it is on the board.
  function ratings(t, members) {
    const mean = t.reduce((s, v) => s + v, 0) / t.length;
    const out = Array.from(t, v => 400 * (v - mean) / Math.LN10 + 1000);
    const cal = ELO.calibration != null ? members.indexOf(ELO_POS.get(ELO.calibration)) : -1;
    if (cal >= 0) { const shift = 1000 - out[cal]; for (let i = 0; i < out.length; i++) out[i] += shift; }
    return out;
  }
  // The two boards over `datasets`: A with the field without B, B with the field without A, and each
  // dataset's win totals on both (the field's, less the wins against the left-out method).
  function boardsFor(datasets, ma = state.a, mb = state.b) {
    const a = ELO_POS.get(ma), b = ELO_POS.get(mb);
    if (a === undefined || b === undefined || a === b) return null;
    const n = ELO.field.length;
    const boardA = [...Array(n).keys()].filter(k => k !== b), boardB = [...Array(n).keys()].filter(k => k !== a);
    const perDataset = datasets.map(d => {
      const full = ELO_WINS[d], vsA = winsAgainst(d, a), vsB = winsAgainst(d, b);
      return { a: boardA.map(k => full[k] - vsB[k]), b: boardB.map(k => full[k] - vsA[k]) };
    });
    // The methods on both boards, by their place on each, and the anchor's place on each (-1 if missing).
    const shared = [...Array(n).keys()].filter(k => k !== a && k !== b);
    const anchor = ELO.calibration != null ? ELO_POS.get(ELO.calibration) : undefined;
    return {
      a: a, b: b, n: n, boardA: boardA, boardB: boardB, perDataset: perDataset,
      sharedA: shared.map(k => boardA.indexOf(k)), sharedB: shared.map(k => boardB.indexOf(k)),
      anchorA: anchor === undefined ? -1 : boardA.indexOf(anchor), anchorB: anchor === undefined ? -1 : boardB.indexOf(anchor),
    };
  }
  // Put both boards on one scale. With the anchor on both, each is already pinned to it, as on the leaderboard. When
  // the anchor is A or B itself, the board without it is shifted so that both boards give the methods they share the
  // same average rating (without an anchor, A's board is shifted to agree with B's).
  function alignBoards(bd, ra, rb) {
    if (bd.anchorA >= 0 && bd.anchorB >= 0) return;
    const mean = (r, idx) => idx.reduce((sum, i) => sum + r[i], 0) / (idx.length || 1);
    const gap = mean(ra, bd.sharedA) - mean(rb, bd.sharedB);
    if (bd.anchorA >= 0) { for (let i = 0; i < rb.length; i++) rb[i] += gap; }
    else { for (let i = 0; i < ra.length; i++) ra[i] -= gap; }
  }
  // Both boards fitted with each dataset counted `counts[i]` times (once each without counts).
  function fitBoards(bd, counts, initA, initB) {
    const m = bd.n - 1;
    const wa = new Float64Array(m), wb = new Float64Array(m);
    let tasks = 0;
    bd.perDataset.forEach((w, i) => {
      const c = counts ? counts[i] : 1;
      if (!c) return;
      tasks += c;
      for (let k = 0; k < m; k++) { wa[k] += c * w.a[k]; wb[k] += c * w.b[k]; }
    });
    const ta = bradleyTerry(wa, tasks, initA), tb = bradleyTerry(wb, tasks, initB);
    const ra = ratings(ta, bd.boardA), rb = ratings(tb, bd.boardB);
    alignBoards(bd, ra, rb);
    return { ta: ta, tb: tb, eloA: ra[bd.boardA.indexOf(bd.a)], eloB: rb[bd.boardB.indexOf(bd.b)] };
  }
  // A's Elo against the field without B, and B's against the field without A, over `datasets`,
  // with the paired bootstrap of their gap: both boards fitted on the same dataset resamples.
  function eloGap(datasets, ma = state.a, mb = state.b, resamples = ELO ? ELO.resamples : 0) {
    if (!ELO || datasets.length < 2) return null;
    const bd = boardsFor(datasets, ma, mb);
    if (!bd) return null;
    const point = fitBoards(bd, null);
    const rand = mulberry32(2026);
    const gaps = new Float64Array(resamples);
    const counts = new Uint32Array(datasets.length);
    for (let r = 0; r < resamples; r++) {
      counts.fill(0);
      for (let i = 0; i < datasets.length; i++) counts[Math.floor(rand() * datasets.length)]++;
      const res = fitBoards(bd, counts, point.ta, point.tb);
      gaps[r] = res.eloA - res.eloB;
    }
    gaps.sort();
    let above = 0;
    for (const v of gaps) if (v > 0) above++;
    return {
      eloA: point.eloA, eloB: point.eloB, gap: point.eloA - point.eloB,
      lo: gaps[Math.floor(0.025 * (gaps.length - 1))], hi: gaps[Math.ceil(0.975 * (gaps.length - 1))],
      above: above / gaps.length, datasets: datasets.length,
    };
  }
  // Each dataset's pull on the gap: the gap over `datasets` less the gap refitted without that
  // dataset. Positive when the dataset moves the gap towards A, negative towards B.
  function eloPull(datasets) {
    if (!ELO || datasets.length < 3) return null;
    const bd = boardsFor(datasets);
    if (!bd) return null;
    const point = fitBoards(bd, null);
    const gap = point.eloA - point.eloB;
    const counts = new Uint32Array(datasets.length).fill(1);
    const pull = new Map();
    datasets.forEach((d, i) => {
      counts[i] = 0;
      const res = fitBoards(bd, counts, point.ta, point.tb);
      pull.set(d, gap - (res.eloA - res.eloB));
      counts[i] = 1;
    });
    return pull;
  }

  // ---------- formatting ----------
  function fmtErr(v) {
    if (v == null || !isFinite(v)) return "\u2014";
    const a = Math.abs(v);
    if (a >= 1000) return v.toFixed(0);
    if (a >= 10) return v.toFixed(2);
    if (a >= 0.1) return v.toFixed(4);
    if (a === 0) return "0";
    return v.toPrecision(3);
  }
  // A ratio as a multiple: x0.50, x1.8, x12.
  function fmtRatio(r) {
    if (r == null || !isFinite(r)) return "\u2014";
    return "\u00d7" + (r >= 10 ? r.toFixed(0) : r >= 1 ? r.toFixed(1) : r.toPrecision(2));
  }
  function fmtHours(seconds) {
    if (seconds == null || !isFinite(seconds)) return "\u2014";
    return seconds >= 3600 ? (seconds / 3600).toFixed(1) + " h" : (seconds / 60).toFixed(0) + " min";
  }
  function fmtInt(v) {
    if (v == null || !isFinite(v)) return "\u2014";
    return Math.round(v).toString().replace(/\B(?=(\d{3})+(?!\d))/g, ",");
  }
  function fmtCount(v) {
    if (v == null || !isFinite(v)) return "\u2014";
    const scaled = (x, unit) => (x < 10 ? x.toFixed(1).replace(/\.0$/, "") : Math.round(x).toString()) + unit;
    if (v < 1000) return Math.round(v).toString();
    if (v < 999500) return scaled(v / 1000, "k");
    return scaled(v / 1e6, "M");
  }
  function fmtSigned(v, decimals, unit) {
    if (v == null || !isFinite(v)) return "\u2014";
    const threshold = 0.5 * Math.pow(10, -decimals);
    const sign = v > threshold ? "+" : v < -threshold ? "\u2212" : "";
    return sign + Math.abs(v).toFixed(decimals) + (unit || "");
  }
  function fmtP(p) {
    if (p == null || !isFinite(p)) return "\u2014";
    if (p < 0.0001) return "<0.0001";
    if (p < 0.01) return p.toFixed(4);
    return p.toFixed(3);
  }
  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  function metricName(ds) { return METRIC_DISPLAY[ds.metric] || ds.metric || "error"; }
  function trainRows(ds) { return ds.train_rows != null ? ds.train_rows : ds.rows; }
  function sizeKeyOf(ds) {
    const rows = trainRows(ds);
    if (rows == null) return null;
    for (const b of SIZE_BUCKETS) if (b.max == null || rows <= b.max) return b.key;
    return SIZE_BUCKETS.length ? SIZE_BUCKETS[SIZE_BUCKETS.length - 1].key : null;
  }
  const SIZE_OF = DATASETS.map(sizeKeyOf);
  function sideName(side) { return METHODS[side === "a" ? state.a : state.b].name; }

  // ---------- the test ----------
  // Two-sided Wilcoxon signed-rank test of paired differences (A minus B). Zero differences are
  // dropped (Wilcoxon's own convention); tied absolute differences share their average rank.
  // Up to EXACT_MAX non-zero differences the statistic's null distribution is enumerated
  // exactly (every sign assignment equally likely, given the ranks as they are, ties included);
  // above it, the normal approximation with the tie correction.
  function signedRank(diffs) {
    const d = diffs.filter(v => v !== 0 && v != null && isFinite(v));
    const n = d.length;
    if (!n) return { n: 0, p: 1 };
    const order = d.map((v, i) => i).sort((i, j) => Math.abs(d[i]) - Math.abs(d[j]));
    const rank2 = new Array(n);   // doubled ranks, so averaged ties stay integers
    for (let i = 0; i < n;) {
      let j = i;
      while (j + 1 < n && Math.abs(d[order[j + 1]]) === Math.abs(d[order[i]])) j++;
      for (let k = i; k <= j; k++) rank2[order[k]] = i + j + 2;
      i = j + 1;
    }
    let tPlus = 0, total = 0, sumSq = 0;
    for (let i = 0; i < n; i++) {
      total += rank2[i];
      sumSq += rank2[i] * rank2[i];
      if (d[i] > 0) tPlus += rank2[i];
    }
    if (n > EXACT_MAX) {
      const mean = total / 2, sd = Math.sqrt(sumSq / 4);
      const z = sd > 0 ? (Math.abs(tPlus - mean) - 1) / sd : 0;   // continuity correction of 1/2 a rank, doubled
      return { n: n, p: Math.min(1, 2 * normalUpper(Math.max(0, z))) };
    }
    let dist = new Float64Array(total + 1);
    dist[0] = 1;
    for (const r of rank2) {
      const next = new Float64Array(total + 1);
      for (let s = 0; s + r <= total; s++) {
        const w = dist[s];
        if (!w) continue;
        next[s] += w * 0.5;
        next[s + r] += w * 0.5;
      }
      dist = next;
    }
    let lower = 0, upper = 0;
    for (let s = 0; s <= total; s++) {
      if (s <= tPlus) lower += dist[s];
      if (s >= tPlus) upper += dist[s];
    }
    return { n: n, p: Math.min(1, 2 * Math.min(lower, upper)) };
  }
  // Upper tail of the standard normal (Abramowitz & Stegun 7.1.26 via erfc), ample for a p-value.
  function normalUpper(z) {
    const t = 1 / (1 + 0.3275911 * z / Math.SQRT2);
    const y = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741 + t * (-1.453152027 + t * 1.061405429))));
    return 0.5 * y * Math.exp(-z * z / 2);
  }
  // Benjamini-Hochberg adjusted p-values, in the order given.
  function bhAdjust(ps) {
    const m = ps.length;
    const order = ps.map((p, i) => i).sort((i, j) => ps[i] - ps[j]);
    const q = new Array(m);
    let running = 1;
    for (let k = m - 1; k >= 0; k--) {
      running = Math.min(running, ps[order[k]] * m / (k + 1));
      q[order[k]] = running;
    }
    return q;
  }
  // A small seeded generator, so the interval does not change from one render to the next.
  function mulberry32(seed) {
    return function () {
      seed |= 0; seed = (seed + 0x6d2b79f5) | 0;
      let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------- state ----------
  const state = {
    a: Math.min(CONFIG.defaultA || 0, METHODS.length - 1),
    b: Math.min(CONFIG.defaultB != null ? CONFIG.defaultB : 1, METHODS.length - 1),
    query: "", task: "all", split: "all", size: "all", balance: "all", kind: "all",
    // Subsets of the "by subset" table by label: "in" keeps only their datasets, "out" takes them out of the view.
    marks: new Map(),
    sort: "p", dir: 1, selected: -1, bh: false,
    yKey: "margin", xKey: "sorted",
    // The selected dataset's split chart: what it plots and in which order the splits run.
    splitY: "e", splitOrder: "split",
  };

  // ---------- the pair, per dataset ----------
  // A's margin over B, signed from A's point of view: how much lower the better mean error is
  // than the worse one, as a fraction of the worse one (the per-dataset browser's margin), so a
  // win and a loss of the same size read the same.
  function marginOf(ea, eb) {
    if (ea == null || eb == null) return null;
    if (ea === eb) return 0;
    if (ea < eb) return eb > 0 ? (1 - ea / eb) * 100 : 0;
    return -(ea > 0 ? (1 - eb / ea) * 100 : 0);
  }
  let PAIR = [];
  function computePair() {
    PAIR = DATASETS.map((ds, d) => {
      const pa = key(d, state.a), pb = key(d, state.b);
      if (!pa || !pb) return null;
      let wins = 0, losses = 0, ties = 0;
      const diffs = [];
      for (let i = 0; i < pa.f.length; i++) {
        const x = pa.f[i], y = pb.f[i];
        if (x == null || y == null) continue;
        if (x < y) wins++; else if (x > y) losses++; else ties++;
        diffs.push(x - y);
      }
      const n = wins + losses + ties;
      const test = signedRank(diffs);
      const tested = n >= MIN_SPLITS;
      return {
        pa: pa, pb: pb, n: n, wins: wins, losses: losses, ties: ties,
        margin: marginOf(pa.e, pb.e),
        dRank: pa.r != null && pb.r != null ? pb.r - pa.r : null,
        dImp: pa.i != null && pb.i != null ? pb.i - pa.i : null,
        logRatio: pa.e > 0 && pb.e > 0 ? Math.log(pa.e / pb.e) : null,
        p: tested ? test.p : null,
        pRaw: test.p,
        q: null,
        lead: pa.e < pb.e ? "a" : pa.e > pb.e ? "b" : null,
        // Mean time per split, A over B (below 1: A faster), and as A's speed-up on a log scale.
        trainRatio: pa.t > 0 && pb.t > 0 ? pa.t / pb.t : null,
        inferRatio: pa.it > 0 && pb.it > 0 ? pa.it / pb.it : null,
        speedTrain: pa.t > 0 && pb.t > 0 ? Math.log10(pb.t / pa.t) : null,
        speedInfer: pa.it > 0 && pb.it > 0 ? Math.log10(pb.it / pa.it) : null,
      };
    });
  }
  // The significance verdict of dataset `d`, under the current adjustment: "a" / "b" for a
  // significant lead, "tie" when the splits cannot tell them apart, "na" when too few splits.
  function verdictOf(d) {
    const x = PAIR[d];
    if (!x) return null;
    if (x.p == null) return "na";
    const p = state.bh ? x.q : x.p;
    return p != null && p < ALPHA && x.lead ? x.lead : "tie";
  }
  function significant(d) { const v = verdictOf(d); return v === "a" || v === "b"; }

  // ---------- filters ----------
  function matchesQuery(ds) {
    if (!state.query) return true;
    const hay = [ds.name, ds.key, ds.domain, ds.source, ds.task].filter(Boolean).join(" ").toLowerCase();
    return hay.includes(state.query);
  }
  function passes(d, over) {
    const f = Object.assign({}, state, over);
    const ds = DATASETS[d];
    if (!matchesQuery(ds)) return false;
    if (f.task === "classification" ? !CLASSIFICATION_TASKS.has(ds.task) : (f.task !== "all" && ds.task !== f.task)) return false;
    if (f.split !== "all" && ds.split !== f.split) return false;
    if (f.size !== "all" && SIZE_OF[d] !== f.size) return false;
    if (f.balance === "extreme" ? !ds.extreme : (f.balance !== "all" && ds.balance !== f.balance)) return false;
    if (f.kind !== "all" && !(ds.kinds || []).includes(f.kind)) return false;
    // The subsets marked in the "by subset" table: a kept one must hold, a subtracted one must not.
    if (state.marks.size) {
      for (const row of allSubsetRows()) {
        const mode = state.marks.get(row.label);
        if (mode && (mode === "in") !== row.keep(d)) return false;
      }
    }
    return true;
  }
  // The datasets in view where both methods ran, with the BH adjustment over the tested ones.
  let inView = [];
  let missingInView = 0;
  function computeView() {
    inView = [];
    missingInView = 0;
    for (let d = 0; d < nD; d++) {
      if (!passes(d, {})) continue;
      if (PAIR[d]) inView.push(d); else missingInView++;
    }
    for (const d of inView) PAIR[d].q = null;
    const tested = inView.filter(d => PAIR[d].p != null);
    const q = bhAdjust(tested.map(d => PAIR[d].p));
    tested.forEach((d, i) => { PAIR[d].q = q[i]; });
  }

  // ---------- summary ----------
  function summarize() {
    const box = document.getElementById("summary");
    const rows = inView.map(d => PAIR[d]);
    if (!rows.length) { box.innerHTML = '<div class="hh-empty">No dataset in view has results for both methods.</div>'; return; }
    let aLead = 0, bLead = 0, even = 0, aSig = 0, bSig = 0, tested = 0, aSigRaw = 0, bSigRaw = 0;
    let splitWins = 0, splitTies = 0, splitN = 0, rankA = 0, rankB = 0, rankN = 0;
    for (const d of inView) {
      const x = PAIR[d];
      if (x.lead === "a") aLead++; else if (x.lead === "b") bLead++; else even++;
      if (x.p != null) {
        tested++;
        if (x.p < ALPHA && x.lead === "a") aSigRaw++;
        if (x.p < ALPHA && x.lead === "b") bSigRaw++;
      }
      const v = verdictOf(d);
      if (v === "a") aSig++;
      if (v === "b") bSig++;
      splitWins += x.wins; splitTies += x.ties; splitN += x.n;
      if (x.pa.r != null && x.pb.r != null) { rankA += x.pa.r; rankB += x.pb.r; rankN++; }
    }
    // The overall error ratio A / B: the geometric mean over datasets of the ratio of the mean
    // errors, with a 95% interval from resampling datasets, and the signed-rank test of the
    // per-dataset log ratios. A dataset where either error is zero has no ratio and is left out.
    const logs = rows.map(x => x.logRatio).filter(v => v != null);
    let ratio = null, lo = null, hi = null, pOverall = null;
    if (logs.length) {
      const mean = arr => arr.reduce((s, v) => s + v, 0) / arr.length;
      ratio = Math.exp(mean(logs));
      if (logs.length > 1) {
        const rand = mulberry32(12345);
        const boots = new Float64Array(RESAMPLES);
        for (let b = 0; b < RESAMPLES; b++) {
          let s = 0;
          for (let k = 0; k < logs.length; k++) s += logs[Math.floor(rand() * logs.length)];
          boots[b] = s / logs.length;
        }
        boots.sort();
        lo = Math.exp(boots[Math.floor(0.025 * (RESAMPLES - 1))]);
        hi = Math.exp(boots[Math.ceil(0.975 * (RESAMPLES - 1))]);
        pOverall = signedRank(logs).p;
      }
    }
    const expected = ALPHA * tested;
    const splitRate = splitN ? (splitWins + 0.5 * splitTies) / splitN : null;
    const tile = (label, value, sub, hint) =>
      `<div class="hh-tile"${hint ? ` title="${escapeHtml(hint)}"` : ""}><div class="hh-tlabel">${label}</div>` +
      `<div class="hh-tvalue">${value}</div><div class="hh-tsub">${sub}</div></div>`;
    const sw = side => `<span class="hh-swatch ${side}"></span>`;
    box.innerHTML =
      `<div class="hh-tile" id="elo-tile"><div class="hh-tlabel">Elo gap A \u2212 B</div>` +
      `<div class="hh-tvalue">\u2026</div><div class="hh-tsub">computing</div></div>` +
      tile("Error ratio A / B", ratio == null ? "\u2014" : ratio.toFixed(4),
        lo == null ? `${logs.length} datasets` :
          `95% interval [${lo.toFixed(4)}, ${hi.toFixed(4)}] \u00b7 p = ${fmtP(pOverall)}`,
        "Geometric mean over the datasets in view of A's mean error divided by B's (below 1: A lower). " +
        `The interval resamples datasets (${RESAMPLES} times); p is a two-sided signed-rank test of the per-dataset log ratios.`) +
      tile("Lower mean error", `${sw("a")}${aLead} <span class="hh-vs">vs</span> ${sw("b")}${bLead}`,
        `of ${rows.length} datasets` + (even ? ` \u00b7 ${even} even` : ""),
        "Datasets where each method has the lower mean error over the splits") +
      tile("Significant", `${sw("a")}${aSig} <span class="hh-vs">vs</span> ${sw("b")}${bSig}`,
        state.bh
          ? `BH-adjusted, of ${tested} testable \u00b7 unadjusted ${aSigRaw} vs ${bSigRaw}`
          : `p < ${ALPHA} of ${tested} testable \u00b7 ~${expected.toFixed(1)} by chance`,
        `Datasets where the splits tell the two apart (${SIG.test || "signed-rank test"}, \u03b1 = ${ALPHA}). ` +
        `Testable: at least ${MIN_SPLITS} splits. With no real difference anywhere, about \u03b1 \u00d7 testable datasets ` +
        "come out significant by chance, split between the two sides.") +
      tile("Split win rate of A", splitRate == null ? "\u2014" : (splitRate * 100).toFixed(1) + "%",
        `${fmtInt(splitWins)} won, ${fmtInt(splitTies)} tied, ${fmtInt(splitN - splitWins - splitTies)} lost of ${fmtInt(splitN)}`,
        "Share of the paired splits where A's error is lower than B's, ties counting half") +
      timeTile() +
      tile("Mean rank in the field", rankN ? `${sw("a")}${(rankA / rankN).toFixed(2)} <span class="hh-vs">vs</span> ${sw("b")}${(rankB / rankN).toFixed(2)}` : "\u2014",
        `over ${rankN} datasets` + (missingInView ? ` \u00b7 ${missingInView} without both left out` : ""),
        "Each method's rank among all methods on a dataset, averaged over its splits and then over the datasets in view");
    scheduleElo();
  }
  // Training and prediction time, A against B: the geometric mean over the datasets in view of the
  // ratio of their mean times per split, and each method's total time over those datasets.
  function timeTile() {
    const geo = key => {
      const logs = inView.map(d => PAIR[d][key]).filter(v => v != null).map(Math.log);
      return logs.length ? Math.exp(logs.reduce((s, v) => s + v, 0) / logs.length) : null;
    };
    const total = side => inView.reduce((s, d) => s + ((PAIR[d][side].t || 0) * PAIR[d].n), 0);
    const train = geo("trainRatio"), infer = geo("inferRatio");
    return `<div class="hh-tile" title="Geometric mean over the datasets in view of A's mean time per split divided by B's (below 1: A faster); totals are each method's training time summed over every split in view">` +
      `<div class="hh-tlabel">Train time A / B</div><div class="hh-tvalue">${fmtRatio(train)}</div>` +
      `<div class="hh-tsub">predict ${fmtRatio(infer)} \u00b7 total training A ${fmtHours(total("pa"))}, B ${fmtHours(total("pb"))}</div></div>`;
  }
  // The Elo gap takes a few hundred milliseconds, so it fills its tile after the rest of the page;
  // a newer request (another pair or filter) supersedes one still waiting.
  let eloTicket = 0;
  const eloCache = new Map();
  function scheduleElo() {
    const ticket = ++eloTicket;
    ELO_PULL = null;
    setTimeout(() => {
      if (ticket !== eloTicket) return;
      const datasets = [];
      for (let d = 0; d < nD; d++) if (passes(d, {})) datasets.push(d);
      const res = eloGapOf(datasets);
      const el = document.getElementById("elo-tile");
      if (!el) return;
      el.title = "The leaderboard's Elo, paired: A against the field without B and B against the field without A, " +
        `fitted on the same ${ELO ? ELO.resamples : 0} resamples of the datasets in view (imputed results included, as on the ` +
        "leaderboard; errors that differ only by floating-point noise tie). The interval is the gap's 2.5th to 97.5th " +
        "percentile over the resamples.";
      if (!res) {
        el.querySelector(".hh-tvalue").textContent = "\u2014";
        el.querySelector(".hh-tsub").textContent = !ELO ? "no Elo field in this page" :
          "needs both methods in the field and two datasets in view";
        return;
      }
      el.querySelector(".hh-tvalue").textContent = fmtSigned(res.gap, 1);
      el.querySelector(".hh-tsub").textContent =
        `95% [${fmtSigned(res.lo, 1)}, ${fmtSigned(res.hi, 1)}] \u00b7 A above B in ${(res.above * 100).toFixed(1)}% ` +
        `of resamples \u00b7 A ${res.eloA.toFixed(1)}, B ${res.eloB.toFixed(1)} over ${res.datasets} datasets`;
      postHeight();
      // Then each dataset's pull on that gap, for the list's column and the selected dataset.
      setTimeout(() => {
        if (ticket !== eloTicket) return;
        const cacheKey = `${state.a}|${state.b}|${datasets.join(",")}`;
        if (!pullCache.has(cacheKey)) pullCache.set(cacheKey, eloPull(datasets));
        ELO_PULL = pullCache.get(cacheKey);
        renderList();
        if (state.yKey === "pull") drawChart();
        const fact = document.getElementById("pull-fact");
        if (fact) fact.textContent = pullText(state.selected);
      }, 0);
    }, 30);
  }
  let ELO_PULL = null;
  const pullCache = new Map();

  // ---------- methods significantly better than A, per dataset ----------
  // For every dataset, the methods whose error is significantly lower than A's there: the per-dataset
  // test of the page (two-sided signed-rank over the paired splits, at least MIN_SPLITS of them), run
  // against A for every measured method, BH-adjusted over those methods when the box is ticked.
  // Independent of B and of the filters, so it is recomputed only when A or the adjustment changes.
  let BETTER = null, betterKey = null, betterTicket = 0;
  function scheduleBetter() {
    const wanted = `${state.a}|${state.bh}`;
    if (wanted === betterKey) return;
    const ticket = ++betterTicket;
    BETTER = null;
    setTimeout(() => {
      if (ticket !== betterTicket) return;
      BETTER = DATASETS.map((ds, d) => {
        const pa = key(d, state.a);
        if (!pa) return null;
        const rows = [];
        for (const p of BY_DATASET[d]) {
          if (p.m === state.a) continue;
          const diffs = [];
          for (let i = 0; i < pa.f.length; i++) if (pa.f[i] != null && p.f[i] != null) diffs.push(pa.f[i] - p.f[i]);
          if (diffs.length >= MIN_SPLITS) rows.push({ m: p.m, p: signedRank(diffs).p, better: p.e < pa.e });
        }
        const ps = state.bh ? bhAdjust(rows.map(r => r.p)) : rows.map(r => r.p);
        const methods = rows.filter((r, i) => ps[i] < ALPHA && r.better).map(r => r.m);
        methods.sort((u, v) => (key(d, u).e - key(d, v).e));
        return { methods: methods, tested: rows.length };
      });
      betterKey = wanted;
      renderList();
      const fact = document.getElementById("better-fact");
      if (fact) fact.textContent = betterText(state.selected);
    }, 60);
  }
  function betterCell(d) {
    const b = BETTER ? BETTER[d] : null;
    if (!BETTER) return '<td class="hh-num hh-nosig">\u2026</td>';
    if (!b || !b.tested) return '<td class="hh-num hh-nosig">\u2014</td>';
    const names = b.methods.map(m => METHODS[m].name);
    const title = names.length ? `Significantly lower error than A: ${names.join(", ")}` : "No method is significantly better than A here";
    return `<td class="hh-num${b.methods.length ? " hh-sig" : " hh-nosig"}" title="${escapeHtml(title)}">${b.methods.length} / ${b.tested}</td>`;
  }
  function betterText(d) {
    if (!BETTER) return "computing\u2026";
    const b = BETTER[d];
    if (!b || !b.tested) return "too few splits to test";
    if (!b.methods.length) return `none of ${b.tested}`;
    const names = b.methods.map(m => METHODS[m].name);
    return `${names.length} of ${b.tested}: ${names.slice(0, 4).join(", ")}${names.length > 4 ? ", \u2026" : ""}`;
  }
  function pullText(d) {
    if (!ELO) return "\u2014";
    if (!ELO_PULL) return "computing\u2026";
    const v = ELO_PULL.get(d);
    return v == null ? "\u2014" : `${fmtSigned(v, 2)} Elo (${v > 0 ? "towards A" : v < 0 ? "towards B" : "none"})`;
  }

  // ---------- the chart ----------
  const Y_AXES = [
    { key: "margin", label: "A's margin over B (%)", fmt: v => fmtSigned(v, 1, "%") },
    { key: "dRank", label: "Rank difference (B's rank \u2212 A's)", fmt: v => fmtSigned(v, 2) },
    { key: "dImp", label: "Improvability difference (B \u2212 A, pp)", fmt: v => fmtSigned(v, 2) },
    // Each dataset's pull on the Elo gap over the datasets in view (the gap less the gap refitted without it); it
    // arrives after the gap itself, and the chart is redrawn then.
    { key: "pull", label: "Pull on the Elo gap (Elo)", fmt: v => fmtSigned(v, 1), value: d => (ELO_PULL ? (ELO_PULL.get(d) ?? null) : null),
      fullRange: true },
    // Time on a log scale, as A's speed-up: up when A is faster. Error significance does not apply.
    { key: "speedTrain", label: "Training speed-up of A (log)", fmt: v => fmtRatio(Math.pow(10, v)), time: true },
    { key: "speedInfer", label: "Prediction speed-up of A (log)", fmt: v => fmtRatio(Math.pow(10, v)), time: true },
  ];
  // The x-axis: the datasets sorted by the y value, or a value of each dataset on a scale of its kind
  // ("log" above 0, "log1p" a log scale that keeps 0, "ratio" a log scale around 1, "percent").
  const X_AXES = [
    { key: "sorted", label: "Datasets, sorted" },
    { key: "rows", label: "Training rows", scale: "log", value: d => trainRows(DATASETS[d]) },
    { key: "features", label: "Features", scale: "log", value: d => DATASETS[d].features },
    // The margin itself: with the pull or the rank difference on the y-axis, a line through 0 fitted on each side
    // shows whether a loss moves the method more than a gain of the same size.
    { key: "margin", label: "A's margin over B (%)", scale: "linear", value: d => PAIR[d].margin, fmt: v => fmtSigned(v, 1, "%"), sides: true },
    { key: "impB", label: "B's distance to the best (improvability, %)", scale: "linear", value: d => PAIR[d].pb.i, fmt: v => v.toFixed(1) + "%" },
    { key: "trainRatio", label: "Training time A / B", scale: "ratio", value: d => PAIR[d].trainRatio },
    { key: "inferRatio", label: "Prediction time A / B", scale: "ratio", value: d => PAIR[d].inferRatio },
    ...(CONFIG.xAxes || []).map(ax => ({
      key: "m-" + ax.key, label: ax.label, scale: ax.scale, value: d => (DATASETS[d].x || {})[ax.key],
    })),
  ];
  const CHART_H = 300;
  const M = { l: 62, r: 16, t: 14, b: 44 };

  function drawChart() {
    const svg = document.getElementById("chart");
    const box = document.getElementById("chartbox");
    svg.textContent = "";
    if (!box.clientWidth) return;
    const W = Math.max(340, Math.round(box.clientWidth));
    const H = CHART_H;
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    const yAxis = Y_AXES.find(a => a.key === state.yKey) || Y_AXES[0];
    const xAxis = X_AXES.find(a => a.key === state.xKey) || X_AXES[0];
    const val = yAxis.value || (d => PAIR[d][yAxis.key]);
    const strong = d => yAxis.time || significant(d);
    let items = inView.filter(d => val(d) != null);
    if (xAxis.key !== "sorted") {
      const positive = xAxis.scale === "log" || xAxis.scale === "ratio";
      items = items.filter(d => { const v = xAxis.value(d); return v != null && isFinite(v) && (!positive || v > 0); });
    }
    if (!items.length) {
      el("text", { x: W / 2, y: H / 2, "text-anchor": "middle", "font-size": 13, fill: "var(--muted)" }, svg).textContent =
        yAxis.key === "pull" && ELO && !ELO_PULL ? "Computing each dataset's pull on the Elo gap\u2026" : "Nothing to plot";
      return;
    }
    // A symmetric axis, clipped near the largest values: one dataset 80% apart would otherwise
    // flatten every other bar into the zero line. Clipped marks get a cap and keep their value in the hover.
    const mags = items.map(d => Math.abs(val(d))).sort((a, b) => a - b);
    const qIdx = Math.min(mags.length - 1, Math.floor(0.97 * (mags.length - 1)));
    // An axis whose extremes are the point (the datasets that drive the Elo gap) shows its full range.
    const limit = yAxis.fullRange ? Math.max(mags[mags.length - 1] * 1.08, 1e-9)
      : Math.max(mags[qIdx] * 1.12, mags[mags.length - 1] <= mags[qIdx] * 1.5 ? mags[mags.length - 1] * 1.05 : 0, 1e-9);
    const y0 = -limit, y1 = limit;
    const Y = v => M.t + (1 - (Math.max(y0, Math.min(y1, v)) - y0) / (y1 - y0)) * (H - M.t - M.b);
    const grid = el("g", {}, svg);
    for (const t of ticks(y0, y1, 6)) {
      const gy = Y(t);
      el("line", { x1: M.l, y1: gy, x2: W - M.r, y2: gy, stroke: "var(--line)", "stroke-width": t === 0 ? 0 : 1 }, grid);
      el("text", { x: M.l - 7, y: gy + 4, "text-anchor": "end", "font-size": 11.5, fill: "var(--muted)" }, grid).textContent =
        yAxis.fmt(t);
    }
    el("line", { x1: M.l, y1: Y(0), x2: W - M.r, y2: Y(0), stroke: "var(--muted)", "stroke-width": 1.2 }, grid);
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": 12.5, "font-weight": 650, fill: "var(--ink)",
      transform: `translate(14 ${(M.t + H - M.b) / 2}) rotate(-90)`,
    }, grid).textContent = yAxis.label;
    // A's bars fill the left of the sorted chart and B's the right, so each label sits over the other side's half.
    el("text", { x: W - M.r - 6, y: M.t + 12, "text-anchor": "end", "font-size": 11.5, fill: "var(--muted)" }, grid).textContent =
      yAxis.time ? "\u25b2 A faster" : "\u25b2 A ahead";
    el("text", { x: M.l + 6, y: H - M.b - 6, "font-size": 11.5, fill: "var(--muted)" }, grid).textContent =
      yAxis.time ? "\u25bc B faster" : "\u25bc B ahead";

    const marks = el("g", {}, svg);
    const hits = el("g", {}, svg);
    const color = d => (val(d) >= 0 ? "var(--side-a)" : "var(--side-b)");
    const place = [];
    if (xAxis.key === "sorted") {
      items.sort((a, b) => val(b) - val(a));
      const step = (W - M.l - M.r) / items.length;
      const bw = Math.max(1, Math.min(14, step - (step > 4 ? 2 : 0.5)));
      items.forEach((d, i) => {
        const cx = M.l + step * (i + 0.5);
        const v = val(d), top = Y(Math.max(v, 0)), bottom = Y(Math.min(v, 0));
        const h = Math.max(1, bottom - top);
        el("rect", {
          x: cx - bw / 2, y: top, width: bw, height: h, rx: Math.min(2, bw / 2), fill: color(d),
          opacity: strong(d) ? 1 : 0.32,
        }, marks);
        if (Math.abs(v) > limit) {
          el("path", {
            d: v > 0 ? `M${cx - 4},${M.t + 5} L${cx},${M.t - 1} L${cx + 4},${M.t + 5}Z` : `M${cx - 4},${H - M.b - 5} L${cx},${H - M.b + 1} L${cx + 4},${H - M.b - 5}Z`,
            fill: "var(--ink)",
          }, marks);
        }
        if (d === state.selected) {
          el("rect", { x: cx - bw / 2 - 2, y: top - 2, width: bw + 4, height: h + 4, rx: 3, fill: "none", stroke: "var(--ink)", "stroke-width": 1.5 }, marks);
        }
        place.push({ d: d, x: cx, w: Math.max(step, 6), y: (top + bottom) / 2, h: Math.max(h, 10) });
      });
      el("text", { x: (M.l + W - M.r) / 2, y: H - 14, "text-anchor": "middle", "font-size": 12.5, "font-weight": 650, fill: "var(--ink)" }, grid)
        .textContent = `${items.length} datasets, sorted by ${yAxis.label.replace(/ \(.*\)$/, "").toLowerCase()}`;
    } else {
      const xv = xAxis.value;
      const xs = items.map(xv);
      const lo = Math.min(...xs), hi = Math.max(...xs);
      // Position on the axis as a fraction of its span, the tick values, and how a tick reads.
      let pos, tickValues, tickText, note;
      if (xAxis.scale === "log1p") {
        const span = Math.log10(1 + hi) * 1.05 || 1;
        pos = v => Math.log10(1 + v) / span;
        tickValues = [0];
        for (let v = 1; v <= hi * 1.05; v *= 10) tickValues.push(v);
        tickText = fmtCount;
        note = " (log scale, 0 kept)";
      } else if (xAxis.scale === "linear") {
        const lo0 = Math.min(lo, 0), hi0 = Math.max(hi, 0), pad = (hi0 - lo0) * 0.05 || 1;
        const a0 = lo0 - pad, a1 = hi0 + pad;
        pos = v => (v - a0) / (a1 - a0);
        tickValues = ticks(a0, a1, 7);
        tickText = xAxis.fmt;
        note = "";
      } else if (xAxis.scale === "percent") {
        const top = hi * 1.08 || 0.01;
        pos = v => v / top;
        tickValues = ticks(0, top, 5);
        tickText = v => (v * 100).toFixed(v * 100 < 1 && v > 0 ? 1 : 0) + "%";
        note = "";
      } else {
        const lx0 = Math.log10(lo * 0.8), lx1 = Math.log10(hi * 1.25);
        pos = v => (Math.log10(v) - lx0) / (lx1 - lx0 || 1);
        if (xAxis.scale === "ratio") {
          tickValues = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100].filter(v => v >= lo * 0.8 && v <= hi * 1.25);
          tickText = fmtRatio;
          note = " (log scale; below \u00d71: A faster)";
        } else {
          tickValues = [];
          for (let e = Math.ceil(lx0); e <= Math.floor(lx1); e++) tickValues.push(Math.pow(10, e));
          tickText = fmtCount;
          note = " (log scale)";
        }
      }
      const X = v => M.l + pos(v) * (W - M.l - M.r);
      for (const t of tickValues) {
        const gx = X(t);
        el("line", {
          x1: gx, y1: M.t, x2: gx, y2: H - M.b, stroke: xAxis.scale === "ratio" && t === 1 ? "var(--muted)" : "var(--line)",
        }, grid);
        el("text", { x: gx, y: H - M.b + 16, "text-anchor": "middle", "font-size": 11.5, fill: "var(--muted)" }, grid).textContent = tickText(t);
      }
      if (xAxis.scale === "linear" && lo < 0 && hi > 0) {
        el("line", { x1: X(0), y1: M.t, x2: X(0), y2: H - M.b, stroke: "var(--muted)", "stroke-width": 1.2 }, grid);
      }
      // One least-squares line through the origin per side of the margin, with its slope.
      if (xAxis.sides && !yAxis.time && yAxis.key !== "margin") {
        for (const [side, keep] of [["a", v => v > 0], ["b", v => v < 0]]) {
          const pts = items.filter(d => keep(xv(d)));
          const sxx = pts.reduce((sum, d) => sum + xv(d) * xv(d), 0);
          if (pts.length < 3 || !sxx) continue;
          const slope = pts.reduce((sum, d) => sum + xv(d) * val(d), 0) / sxx;
          const end = side === "a" ? Math.max(...pts.map(xv)) : Math.min(...pts.map(xv));
          el("line", {
            x1: X(0), y1: Y(0), x2: X(end), y2: Y(slope * end), stroke: `var(--side-${side})`, "stroke-width": 2,
            "stroke-dasharray": "7 4", opacity: 0.9,
          }, grid);
          el("text", {
            x: X(end) + (side === "a" ? -4 : 4), y: Math.max(M.t + 12, Math.min(H - M.b - 6, Y(slope * end) + (side === "a" ? -8 : 14))),
            "text-anchor": side === "a" ? "end" : "start", "font-size": 11.5, "font-weight": 650, fill: "var(--ink)",
          }, grid).textContent = `${fmtSigned(slope, Math.abs(slope) < 1 ? 3 : 2)} per 1% margin (${pts.length} datasets)`;
        }
      }
      el("text", { x: (M.l + W - M.r) / 2, y: H - 8, "text-anchor": "middle", "font-size": 12.5, "font-weight": 650, fill: "var(--ink)" }, grid)
        .textContent = `${xAxis.label}${note} \u00b7 ${items.length} datasets`;
      for (const d of items) {
        const cx = X(xv(d)), cy = Y(val(d));
        const sig = strong(d);
        el("circle", {
          cx: cx, cy: cy, r: d === state.selected ? 6 : 4.5, fill: color(d), opacity: sig ? 1 : 0.35,
          stroke: d === state.selected ? "var(--ink)" : "var(--card)", "stroke-width": d === state.selected ? 1.8 : 1,
        }, marks);
        place.push({ d: d, x: cx, w: 12, y: cy, h: 12 });
      }
    }
    for (const pl of place) {
      const hit = el("rect", { x: pl.x - pl.w / 2, y: Math.min(pl.y - pl.h / 2, H - M.b), width: pl.w, height: pl.h, fill: "transparent", cursor: "pointer" }, hits);
      if (xAxis.key === "sorted") { hit.setAttribute("y", M.t); hit.setAttribute("height", H - M.t - M.b); }
      hit.addEventListener("mouseenter", ev => tip.show(datasetTip(pl.d), ev));
      hit.addEventListener("mousemove", ev => tip.move(ev));
      hit.addEventListener("mouseleave", () => tip.hide());
      hit.addEventListener("click", () => select(pl.d));
    }
    writeHash();
    postHeight();
  }

  function verdictText(d) {
    const v = verdictOf(d), x = PAIR[d];
    if (v === "na") return `too few splits to test (${x.n}, needs ${MIN_SPLITS})`;
    const p = state.bh ? `q = ${fmtP(x.q)}` : `p = ${fmtP(x.p)}`;
    if (v === "a") return `A significantly lower (${p})`;
    if (v === "b") return `B significantly lower (${p})`;
    return `not significant (${p})`;
  }
  function datasetTip(d) {
    const ds = DATASETS[d], x = PAIR[d];
    const pull = ELO_PULL ? ELO_PULL.get(d) : null;
    return `<div class="t-name">${escapeHtml(ds.name)}</div>` +
      `<div>${escapeHtml(metricName(ds))}: A <b>${fmtErr(x.pa.e)}</b> \u00b7 B <b>${fmtErr(x.pb.e)}</b></div>` +
      `<div>A's margin: <b>${fmtSigned(x.margin, 2, "%")}</b></div>` +
      `<div>Rank: A ${fmtNum(x.pa.r, 1)} \u00b7 B ${fmtNum(x.pb.r, 1)}</div>` +
      `<div>Splits: A lower on ${x.wins}, B on ${x.losses}${x.ties ? `, tied ${x.ties}` : ""}</div>` +
      `<div>${escapeHtml(verdictText(d))}</div>` +
      (pull != null ? `<div>Pull on the Elo gap: <b>${fmtSigned(pull, 2)}</b></div>` : "") +
      (x.trainRatio != null ? `<div>Time A / B: train ${fmtRatio(x.trainRatio)}, predict ${fmtRatio(x.inferRatio)}</div>` : "");
  }

  function buildLegend() {
    const sq = (fill, opacity) => `<svg width="12" height="12" viewBox="0 0 12 12"><rect x="1" y="1" width="10" height="10" rx="2" fill="${fill}" opacity="${opacity}"/></svg>`;
    document.getElementById("legendstrip").innerHTML =
      `<span class="item">${sq("var(--side-a)", 1)} <span>A ahead: ${escapeHtml(sideName("a"))}</span></span>` +
      `<span class="item">${sq("var(--side-b)", 1)} <span>B ahead: ${escapeHtml(sideName("b"))}</span></span>` +
      `<span class="item">${sq("var(--muted)", 0.32)} <span>Faded: not significant${state.bh ? " after BH" : ""} (\u03b1 = ${ALPHA}) or too few splits</span></span>`;
  }

  // ---------- the list ----------
  const COLUMNS = [
    { key: "pos", label: "#", sortable: false },
    { key: "name", label: "Dataset" },
    { key: "task", label: "Task", wide: true },
    { key: "train_rows", label: "Rows", num: true, hint: "Training rows: the largest training split" },
    { key: "features", label: "Feat.", num: true, wide: true },
    { key: "margin", label: "Margin", num: true, hint: "A's margin over B on the mean error: positive when A's error is lower" },
    { key: "rankA", label: "Rank A", num: true, wide: true, hint: "A's mean rank among all methods on this dataset" },
    { key: "rankB", label: "Rank B", num: true, wide: true, hint: "B's mean rank among all methods on this dataset" },
    { key: "dRank", label: "\u0394 rank", num: true, hint: "B's rank minus A's: positive when A ranks better" },
    { key: "pull", label: "Elo pull", num: true, hint: "How much this dataset moves the Elo gap A \u2212 B over the datasets in view: the gap less the gap refitted without it (positive: towards A)" },
    { key: "splits", label: "Splits A:B", num: true, mid: true, hint: "Splits where A's error is lower : splits where B's is (ties left out)" },
    { key: "p", label: "p", num: true, mid: true, hint: "Two-sided Wilcoxon signed-rank test over the paired splits (BH-adjusted when the box is ticked); sorts most significant first" },
    { key: "verdict", label: "Verdict", hint: "Significant leads first, A's strongest at the top in one direction and B's in the other" },
    { key: "better", label: "Better than A", num: true, hint: "Methods whose error is significantly lower than A's on this dataset (two-sided signed-rank test over the splits, \u03b1 = 0.05; BH over the methods when the box is ticked), of the methods tested here" },
    { key: "train", label: "Train A/B", num: true, wide: true, hint: "A's mean training time per split divided by B's (below \u00d71: A faster)" },
  ];
  function sortValue(d, sortKey) {
    const ds = DATASETS[d], x = PAIR[d];
    switch (sortKey) {
      case "name": return (ds.name || "").toLowerCase();
      case "task": return ds.task || "";
      case "train_rows": return trainRows(ds);
      case "features": return ds.features;
      case "margin": return x.margin;
      case "rankA": return x.pa.r;
      case "rankB": return x.pb.r;
      case "dRank": return x.dRank;
      case "pull": return ELO_PULL ? (ELO_PULL.get(d) ?? null) : null;
      case "train": return x.trainRatio;
      case "better": return BETTER && BETTER[d] ? BETTER[d].methods.length : null;
      case "splits": return x.n ? (x.wins - x.losses) / x.n : null;
      case "p": return state.bh ? x.q : x.p;
      case "verdict": {
        // Signed strength: -log10(p) towards A positive, towards B negative; untested last.
        const p = state.bh ? x.q : x.p;
        if (p == null) return null;
        const s = -Math.log10(Math.max(p, 1e-300));
        return x.lead === "a" ? s : x.lead === "b" ? -s : 0;
      }
      default: return null;
    }
  }
  // Natural first directions: most significant and largest first.
  const DESCENDING_FIRST = new Set(["train_rows", "features", "verdict", "better"]);
  function visibleOrder() {
    const out = inView.slice();
    const compare = (a, b, sortKey, direction) => {
      const va = sortValue(a, sortKey), vb = sortValue(b, sortKey);
      if (va == null && vb == null) return 0;
      if (va == null) return 1;
      if (vb == null) return -1;
      if (typeof va === "string") return direction * va.localeCompare(vb);
      return direction * (va - vb);
    };
    const absMargin = d => -Math.abs(PAIR[d].margin || 0);
    out.sort((a, b) => compare(a, b, state.sort, state.dir) || absMargin(a) - absMargin(b) || compare(a, b, "name", 1));
    return out;
  }
  let visible = [];
  function buildHead() {
    let html = "<tr>";
    for (const col of COLUMNS) {
      const sortable = col.sortable !== false;
      const cls = (col.num ? "hh-num " : "") + (col.wide ? "hh-wide " : "") + (col.mid ? "hh-mid " : "") + (sortable ? "sortable" : "");
      html += `<th class="${cls}" data-k="${col.key}"${col.hint ? ` title="${escapeHtml(col.hint)}"` : ""}>${col.label}</th>`;
    }
    document.getElementById("thead").innerHTML = html + "</tr>";
    for (const th of document.querySelectorAll("#thead th.sortable")) {
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (state.sort === k) state.dir = -state.dir;
        else { state.sort = k; state.dir = DESCENDING_FIRST.has(k) ? -1 : 1; }
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
  // A signed value with the mark of the side it favours (positive: A).
  function leadCell(value, text, sig) {
    const side = value > 0 ? "a" : value < 0 ? "b" : "";
    return `<td class="hh-num"><span class="hh-lead ${side}${sig ? " hh-sig" : ""}">${text}</span></td>`;
  }
  function renderList() {
    visible = visibleOrder();
    syncHead();
    const tbody = document.getElementById("tbody");
    const frag = document.createDocumentFragment();
    visible.forEach((d, i) => {
      const ds = DATASETS[d], x = PAIR[d];
      const v = verdictOf(d), sig = v === "a" || v === "b";
      const tr = document.createElement("tr");
      tr.dataset.d = String(d);
      if (d === state.selected) tr.className = "sel";
      tr.title = [ds.name, `${fmtInt(trainRows(ds))} train rows \u00b7 ${fmtInt(ds.features)} features \u00b7 ${x.n} splits`,
        [ds.domain, ds.source, ds.year].filter(Boolean).join(" \u00b7 ")].filter(Boolean).join("\n");
      const p = state.bh ? x.q : x.p;
      const verdictLabel = { a: "A lower", b: "B lower", tie: "no difference", na: "too few splits" }[v];
      tr.innerHTML =
        `<td class="hh-pos">${i + 1}</td>` +
        `<td><span class="hh-name">${escapeHtml(ds.name)}</span></td>` +
        `<td class="hh-wide"><span class="hh-tag">${escapeHtml(TASK_SHORT[ds.task] || ds.task || "?")}</span></td>` +
        `<td class="hh-num">${fmtCount(trainRows(ds))}</td>` +
        `<td class="hh-num hh-wide">${fmtCount(ds.features)}</td>` +
        leadCell(x.margin, fmtSigned(x.margin, 2, "%"), sig) +
        `<td class="hh-num hh-wide">${fmtNum(x.pa.r, 1)}</td>` +
        `<td class="hh-num hh-wide">${fmtNum(x.pb.r, 1)}</td>` +
        leadCell(x.dRank, fmtSigned(x.dRank, 1), false) +
        (() => {
          const pull = ELO_PULL ? ELO_PULL.get(d) : null;
          return pull != null ? leadCell(pull, fmtSigned(pull, 2), Math.abs(pull) >= 1) :
            `<td class="hh-num hh-nosig">${ELO && !ELO_PULL ? "\u2026" : "\u2014"}</td>`;
        })() +
        `<td class="hh-num hh-mid">${x.wins}:${x.losses}${x.ties ? ` <span class="hh-nosig">(${x.ties}=)</span>` : ""}</td>` +
        `<td class="hh-num hh-mid ${sig ? "hh-sig" : "hh-nosig"}">${p == null ? "\u2014" : fmtP(p)}</td>` +
        `<td><span class="hh-verdict ${v}" title="${escapeHtml(verdictText(d))}">${verdictLabel}</span></td>` +
        betterCell(d) +
        (x.trainRatio != null ? leadCell(-x.speedTrain, fmtRatio(x.trainRatio), false).replace('class="hh-num"', 'class="hh-num hh-wide"')
          : '<td class="hh-num hh-wide">\u2014</td>');
      tr.addEventListener("click", () => select(d));
      frag.appendChild(tr);
    });
    tbody.textContent = "";
    tbody.appendChild(frag);
    let shown = 0;
    for (let d = 0; d < nD; d++) if (passes(d, {})) shown++;
    document.getElementById("count").textContent =
      `${inView.length} datasets with both methods` + (shown - inView.length ? ` (${shown - inView.length} in view without both)` : "") +
      ` of ${nD}. Click a row for its splits; \u2190 \u2192 step through the list.`;
    syncFilterCounts();
    writeHash();
    if (visible.length && !visible.includes(state.selected)) select(visible[0], { scroll: false });
    else postHeight();
  }

  // ---------- one dataset ----------
  function select(d, options) {
    const opts = options || {};
    state.selected = d;
    for (const tr of document.getElementById("tbody").children) tr.classList.toggle("sel", Number(tr.dataset.d) === d);
    if (opts.scroll !== false) {
      const row = [...document.getElementById("tbody").children].find(tr => Number(tr.dataset.d) === d);
      if (row) row.scrollIntoView({ block: "nearest" });
    }
    renderDetail();
    drawChart();
  }
  function step(delta) {
    if (!visible.length) return;
    const at = visible.indexOf(state.selected);
    const next = at < 0 ? 0 : Math.min(visible.length - 1, Math.max(0, at + delta));
    select(visible[next]);
  }
  function renderDetail() {
    const box = document.getElementById("detail");
    const d = state.selected;
    if (d < 0 || !PAIR[d]) { box.innerHTML = '<div class="hh-empty">Pick a dataset in the list or the chart.</div>'; return; }
    const ds = DATASETS[d], x = PAIR[d];
    const split = ds.split ? ` \u00b7 ${SPLIT_LABELS[ds.split] || ds.split} splits` : "";
    const facts = [
      ["A: " + sideName("a"), `${fmtErr(x.pa.e)}${x.pa.s != null ? ` \u00b1 ${fmtErr(x.pa.s)}` : ""}`],
      ["B: " + sideName("b"), `${fmtErr(x.pb.e)}${x.pb.s != null ? ` \u00b1 ${fmtErr(x.pb.s)}` : ""}`],
      ["A's margin over B", fmtSigned(x.margin, 2, "%")],
      ["Rank in the field, A / B", `${fmtNum(x.pa.r, 2)} / ${fmtNum(x.pb.r, 2)}`],
      ["Splits A lower : B lower : tied", `${x.wins} : ${x.losses} : ${x.ties}`],
      ["Signed-rank test", x.p == null ? `too few splits (${x.n})` : `p = ${fmtP(x.p)}${x.q != null ? `, BH q = ${fmtP(x.q)}` : ""}`],
      ["Verdict", verdictText(d)],
      ["Pull on the Elo gap", pullText(d)],
      ["Methods significantly better than A", betterText(d)],
      ["Mean fit time, A / B", `${x.pa.t != null ? fmtTime(x.pa.t) : "\u2014"} / ${x.pb.t != null ? fmtTime(x.pb.t) : "\u2014"} (${fmtRatio(x.trainRatio)})`],
      ["Mean predict time, A / B", `${x.pa.it != null ? fmtTime(x.pa.it) : "\u2014"} / ${x.pb.it != null ? fmtTime(x.pb.it) : "\u2014"} (${fmtRatio(x.inferRatio)})`],
    ];
    box.innerHTML =
      `<div class="hh-dhead"><span class="hh-dname">${escapeHtml(ds.name)}</span>` +
      `<span class="hh-dsub">${escapeHtml(TASK_LABELS[ds.task] || ds.task || "")} \u00b7 ${escapeHtml(metricName(ds))} \u00b7 ` +
      `${fmtInt(trainRows(ds))} rows \u00d7 ${fmtInt(ds.features)} features \u00b7 ${x.n} splits${split}</span></div>` +
      `<dl class="hh-facts">${facts.map(([k, v]) => `<div class="hh-fact"><dt title="${escapeHtml(k)}">${escapeHtml(k.length > 40 ? k.slice(0, 39) + "\u2026" : k)}</dt><dd${k === "Pull on the Elo gap" ? ' id="pull-fact"' : k === "Methods significantly better than A" ? ' id="better-fact"' : ""}>${escapeHtml(v)}</dd></div>`).join("")}</dl>` +
      '<div class="hh-splitbar">' +
      '<label class="metricpick">Y-axis <select id="split-y"></select></label>' +
      '<label class="metricpick">Order <select id="split-order"></select></label></div>' +
      '<div class="chartbox" id="splitbox"><svg id="splitchart" role="img" aria-label="Both methods on every split"></svg></div>' +
      (ds.facts && ds.facts.length
        ? '<details class="hh-meta" open><summary>Dataset metadata</summary><dl class="hh-metagrid">' +
          ds.facts.map(([k, v]) => `<div class="hh-fact"><dt>${escapeHtml(k)}</dt><dd title="${escapeHtml(v)}">${escapeHtml(v)}</dd></div>`).join("") +
          "</dl></details>"
        : "");
    for (const [id, list, field] of [["split-y", SPLIT_Y, "splitY"], ["split-order", SPLIT_ORDER, "splitOrder"]]) {
      const select = document.getElementById(id);
      for (const item of list) {
        const opt = document.createElement("option");
        opt.value = item.key;
        opt.textContent = typeof item.label === "function" ? item.label(ds) : item.label;
        select.appendChild(opt);
      }
      select.value = state[field];
      select.addEventListener("change", ev => { state[field] = ev.target.value; drawSplits(); });
    }
    drawSplits();
    postHeight();
  }

  // ---------- one dataset's splits ----------
  // Every measured method of a dataset, for the per-split rank and improvability: the same field,
  // and the same definitions, as the summary's rank and improvability (one split at a time here).
  const BY_DATASET = Array.from({ length: nD }, () => []);
  for (const p of POINTS) BY_DATASET[p.d].push(p);
  const SPLIT_Y = [
    { key: "e", label: ds => `Test error (${metricName(ds)})`, fmt: fmtErr },
    { key: "i", label: "Improvability (%)", fmt: v => fmtNum(v, Math.abs(v) >= 10 ? 1 : 2) },
    { key: "r", label: "Rank in the field (1 = best)", fmt: v => fmtNum(v, Number.isInteger(v) ? 0 : 1) },
  ];
  const SPLIT_ORDER = [
    { key: "split", label: "By split" },
    { key: "gap", label: "Biggest gap first" },
    { key: "a", label: "Highest A first" },
    { key: "b", label: "Highest B first" },
  ];
  // Both methods on split `i` of dataset `d`: error, improvability (how far behind the split's best,
  // in %) and rank among the whole field (average rank for ties).
  function splitValues(d, i, pa, pb) {
    const field = BY_DATASET[d].map(p => p.f[i]).filter(v => v != null);
    const best = Math.min(...field);
    const rank = v => {
      let less = 0, equal = 0;
      for (const u of field) { if (u < v) less++; else if (u === v) equal++; }
      return less + (equal + 1) / 2;
    };
    const imp = v => (v === 0 ? 0 : (1 - best / v) * 100);
    const a = pa.f[i], b = pb.f[i];
    return { a: { e: a, i: imp(a), r: rank(a) }, b: { e: b, i: imp(b), r: rank(b) } };
  }
  // Each split's two values as a pair of dots joined by a line in the colour of the lower error,
  // with each method's mean as a dashed line. The splits run in their own order or by a value.
  function drawSplits() {
    const d = state.selected;
    const svg = document.getElementById("splitchart"), box = document.getElementById("splitbox");
    if (!svg || d < 0 || !PAIR[d] || !box.clientWidth) return;
    svg.textContent = "";
    const x = PAIR[d], ds = DATASETS[d];
    const yAxis = SPLIT_Y.find(m => m.key === state.splitY) || SPLIT_Y[0];
    const order = SPLIT_ORDER.find(o => o.key === state.splitOrder) || SPLIT_ORDER[0];
    const rows = [];
    for (let i = 0; i < x.pa.f.length; i++) {
      if (x.pa.f[i] == null || x.pb.f[i] == null) continue;
      const v = splitValues(d, i, x.pa, x.pb);
      rows.push({ i: i, fold: (ds.folds || [])[i] != null ? ds.folds[i] : i, v: v, ya: v.a[yAxis.key], yb: v.b[yAxis.key] });
    }
    const n = rows.length;
    if (!n) return;
    if (order.key === "gap") rows.sort((p, q) => Math.abs(q.ya - q.yb) - Math.abs(p.ya - p.yb) || p.i - q.i);
    else if (order.key === "a") rows.sort((p, q) => q.ya - p.ya || p.i - q.i);
    else if (order.key === "b") rows.sort((p, q) => q.yb - p.yb || p.i - q.i);
    const W = Math.max(300, Math.round(box.clientWidth)), H = 220;
    const S = { l: 72, r: 12, t: 10, b: 36 };
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    const vals = rows.flatMap(r => [r.ya, r.yb]);
    let lo = Math.min(...vals), hi = Math.max(...vals);
    const pad = (hi - lo) * 0.08 || Math.abs(hi) * 0.05 || 1;
    lo -= pad; hi += pad;
    const X = j => S.l + (n === 1 ? 0.5 : (j + 0.5) / n) * (W - S.l - S.r);
    const Y = v => S.t + (1 - (v - lo) / (hi - lo)) * (H - S.t - S.b);
    const grid = el("g", {}, svg);
    for (const t of ticks(lo, hi, 5)) {
      el("line", { x1: S.l, y1: Y(t), x2: W - S.r, y2: Y(t), stroke: "var(--line)" }, grid);
      el("text", { x: S.l - 6, y: Y(t) + 4, "text-anchor": "end", "font-size": 11, fill: "var(--muted)" }, grid).textContent = yAxis.fmt(t);
    }
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": 12, "font-weight": 650, fill: "var(--ink)",
      transform: `translate(13 ${(S.t + H - S.b) / 2}) rotate(-90)`,
    }, grid).textContent = typeof yAxis.label === "function" ? yAxis.label(ds) : yAxis.label;
    el("text", { x: (S.l + W - S.r) / 2, y: H - 6, "text-anchor": "middle", "font-size": 12, "font-weight": 650, fill: "var(--ink)" }, grid)
      .textContent = order.key === "split" ? "Split" : `Split, ${order.label.toLowerCase()}`;
    const labelEvery = Math.max(1, Math.ceil(n / 25));
    rows.forEach((r, j) => {
      if (j % labelEvery) return;
      el("text", { x: X(j), y: H - S.b + 14, "text-anchor": "middle", "font-size": 10.5, fill: "var(--muted)" }, grid).textContent =
        String(r.fold);
    });
    const mean = side => rows.reduce((s, r) => s + (side === "a" ? r.ya : r.yb), 0) / n;
    for (const side of ["a", "b"]) {
      el("line", {
        x1: S.l, y1: Y(mean(side)), x2: W - S.r, y2: Y(mean(side)), stroke: `var(--side-${side})`, "stroke-width": 1.3,
        "stroke-dasharray": "6 4", opacity: 0.8,
      }, grid);
    }
    const marks = el("g", {}, svg);
    rows.forEach((r, j) => {
      const ea = r.v.a.e, eb = r.v.b.e;
      const winner = ea < eb ? "a" : eb < ea ? "b" : null;
      el("line", {
        x1: X(j), y1: Y(r.ya), x2: X(j), y2: Y(r.yb), stroke: winner ? `var(--side-${winner})` : "var(--muted)", "stroke-width": 2.2, opacity: 0.55,
      }, marks);
      for (const [side, value] of [["b", r.yb], ["a", r.ya]]) {
        el("circle", { cx: X(j), cy: Y(value), r: 4.2, fill: `var(--side-${side})`, stroke: "var(--card)", "stroke-width": 1.2 }, marks);
      }
      const hit = el("rect", { x: X(j) - 9, y: S.t, width: 18, height: H - S.t - S.b, fill: "transparent" }, marks);
      const line = (label, side) =>
        `<div>${label}: <b>${fmtErr(r.v[side].e)}</b> \u00b7 ${fmtNum(r.v[side].i, 2)}% behind the best \u00b7 rank ${fmtNum(r.v[side].r, 1)}</div>`;
      hit.addEventListener("mouseenter", ev => tip.show(
        `<div class="t-name">Split ${r.fold}</div>` + line("A", "a") + line("B", "b") +
        `<div>${winner ? (winner === "a" ? "A" : "B") + " lower by " + fmtSigned(Math.abs(marginOf(ea, eb)), 2, "%").replace("+", "") : "Identical errors"}</div>`, ev));
      hit.addEventListener("mousemove", ev => tip.move(ev));
      hit.addEventListener("mouseleave", () => tip.hide());
    });
    writeHash();
  }

  // ---------- controls ----------
  function buildPickers() {
    const compared = new Set(POINTS.map(p => p.m));
    const order = METHODS.map((m, i) => i).filter(i => compared.has(i)).sort((a, b) => {
      const fa = FAM_ORDER.indexOf(METHODS[a].family), fb = FAM_ORDER.indexOf(METHODS[b].family);
      return fa - fb || METHODS[a].name.localeCompare(METHODS[b].name);
    });
    for (const [id, side] of [["pick-a", "a"], ["pick-b", "b"]]) {
      const select = document.getElementById(id);
      select.textContent = "";
      let family = null, group = null;
      for (const i of order) {
        if (METHODS[i].family !== family) {
          family = METHODS[i].family;
          group = document.createElement("optgroup");
          group.label = family;
          select.appendChild(group);
        }
        const opt = document.createElement("option");
        opt.value = String(i);
        opt.textContent = METHODS[i].name;
        group.appendChild(opt);
      }
      select.value = String(state[side]);
      select.addEventListener("change", ev => { state[side] = Number(ev.target.value); refresh(); });
    }
  }
  document.getElementById("btn-swap").addEventListener("click", () => {
    [state.a, state.b] = [state.b, state.a];
    document.getElementById("pick-a").value = String(state.a);
    document.getElementById("pick-b").value = String(state.b);
    refresh();
  });
  document.getElementById("bh").addEventListener("change", ev => { state.bh = ev.target.checked; refreshView(); });
  for (const [id, list, field] of [["yaxis", Y_AXES, "yKey"], ["xaxis", X_AXES, "xKey"]]) {
    const select = document.getElementById(id);
    for (const item of list) {
      const opt = document.createElement("option");
      opt.value = item.key;
      opt.textContent = item.label;
      select.appendChild(opt);
    }
    select.value = state[field];
    select.addEventListener("change", ev => { state[field] = ev.target.value; drawChart(); });
  }

  function buildFilter(elementId, label, options, field) {
    const box = document.getElementById(elementId);
    if (options.length < 3) { box.hidden = true; return; }
    box.innerHTML = `<span class="grouplabel">${label}</span>`;
    for (const opt of options) {
      const button = document.createElement("button");
      button.className = "hh-fchip";
      button.type = "button";
      button.innerHTML = escapeHtml(opt.label) + '<span class="n"></span>';
      button.dataset.v = opt.key;
      button.dataset.f = field;
      button.addEventListener("click", () => {
        if (button.getAttribute("aria-disabled") === "true") return;
        state[field] = opt.key;
        for (const b of box.querySelectorAll(".hh-fchip")) b.setAttribute("aria-pressed", String(b.dataset.v === state[field]));
        refreshView();
      });
      button.setAttribute("aria-pressed", String(opt.key === state[field]));
      box.appendChild(button);
    }
  }
  function syncFilterCounts() {
    for (const button of document.querySelectorAll(".hh-fchip")) {
      let n = 0;
      for (let d = 0; d < nD; d++) if (PAIR[d] && passes(d, { [button.dataset.f]: button.dataset.v })) n++;
      button.querySelector(".n").textContent = n;
      if (n === 0 && button.dataset.v !== state[button.dataset.f]) button.setAttribute("aria-disabled", "true");
      else button.removeAttribute("aria-disabled");
    }
  }
  const tasksPresent = [...new Set(DATASETS.map(ds => ds.task).filter(Boolean))].sort();
  const bothClassifications = [...CLASSIFICATION_TASKS].every(t => tasksPresent.includes(t));
  buildFilter("taskfilter", "Task", [
    { key: "all", label: "All" },
    ...(bothClassifications ? [{ key: "classification", label: "Classification" }] : []),
    ...tasksPresent.map(t => ({ key: t, label: TASK_LABELS[t] || t })),
  ], "task");
  buildFilter("splitfilter", "Split", [
    { key: "all", label: "All" },
    ...Object.keys(SPLIT_LABELS).filter(k => DATASETS.some(ds => ds.split === k)).map(k => ({ key: k, label: SPLIT_LABELS[k] })),
  ], "split");
  buildFilter("sizefilter", "Size", [
    { key: "all", label: "All" },
    ...SIZE_BUCKETS.filter(b => SIZE_OF.includes(b.key)).map(b => ({ key: b.key, label: b.label })),
  ], "size");
  const BALANCE_LABELS = { balanced: "Balanced", imbalanced: "Imbalanced", extreme: "Extreme" };
  buildFilter("balancefilter", "Target", [
    { key: "all", label: "All" },
    ...Object.keys(BALANCE_LABELS).filter(k => (k === "extreme" ? DATASETS.some(ds => ds.extreme) : DATASETS.some(ds => ds.balance === k)))
      .map(k => ({ key: k, label: BALANCE_LABELS[k] })),
  ], "balance");
  // The datasets' column kinds, as the arena's subsets of the same names draw them; a dataset can be of several.
  const KIND_LABELS = {
    text: "Text", "high-cardinality": "High-card categorical", datetime: "Date", categorical: "Categorical",
    numerical: "Numerical only",
  };
  buildFilter("kindfilter", "Columns", [
    { key: "all", label: "All" },
    ...Object.keys(KIND_LABELS).filter(k => DATASETS.some(ds => (ds.kinds || []).includes(k)))
      .map(k => ({ key: k, label: KIND_LABELS[k] })),
  ], "kind");
  let searchTimer = null;
  document.getElementById("search").addEventListener("input", ev => {
    const value = ev.target.value.trim().toLowerCase();
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.query = value; refreshView(); }, 110);
  });

  // The whole view is kept in the address (the pair, the filters, the subset marks, the sorting,
  // the axes and the selected dataset), so a link opens on the same view. Defaults are left out.
  const FILTER_FIELDS = ["task", "split", "size", "balance", "kind"];
  function writeHash() {
    try {
      const params = new URLSearchParams();
      params.set("a", METHODS[state.a].name);
      params.set("b", METHODS[state.b].name);
      for (const f of FILTER_FIELDS) if (state[f] !== "all") params.set(f, state[f]);
      if (state.query) params.set("q", state.query);
      if (state.marks.size) params.set("marks", [...state.marks].map(([label, mode]) => `${mode}:${label}`).join("|"));
      if (state.sort !== "p" || state.dir !== 1) params.set("sort", `${state.sort}:${state.dir}`);
      if (state.bh) params.set("bh", "1");
      if (state.yKey !== "margin") params.set("y", state.yKey);
      if (state.xKey !== "sorted") params.set("x", state.xKey);
      if (state.splitY !== "e") params.set("sy", state.splitY);
      if (state.splitOrder !== "split") params.set("so", state.splitOrder);
      if (state.selected >= 0 && DATASETS[state.selected]) params.set("ds", DATASETS[state.selected].key);
      const hash = "#" + params.toString();
      if (location.hash !== hash) history.replaceState(null, "", hash);
    } catch (err) { /* a sandboxed frame may refuse; the page works without it */ }
  }
  // Values the page does not offer are ignored, so an old or edited link still opens.
  function readHash() {
    try {
      const params = new URLSearchParams(location.hash.slice(1));
      const byName = new Map(METHODS.map((m, i) => [m.name, i]));
      if (byName.has(params.get("a"))) state.a = byName.get(params.get("a"));
      if (byName.has(params.get("b"))) state.b = byName.get(params.get("b"));
      for (const f of FILTER_FIELDS) {
        const v = params.get(f);
        if (v && [...document.querySelectorAll(".hh-fchip")].some(b => b.dataset.f === f && b.dataset.v === v)) state[f] = v;
      }
      if (params.get("q")) state.query = params.get("q").toLowerCase();
      const labels = new Set(allSubsetRows().map(r => r.label));
      for (const part of (params.get("marks") || "").split("|")) {
        const at = part.indexOf(":");
        const mode = part.slice(0, at), label = part.slice(at + 1);
        if ((mode === "in" || mode === "out") && labels.has(label)) state.marks.set(label, mode);
      }
      const [sortKey, dir] = (params.get("sort") || "").split(":");
      if (COLUMNS.some(c => c.key === sortKey && c.sortable !== false)) { state.sort = sortKey; state.dir = dir === "-1" ? -1 : 1; }
      state.bh = params.get("bh") === "1";
      if (Y_AXES.some(a => a.key === params.get("y"))) state.yKey = params.get("y");
      if (X_AXES.some(a => a.key === params.get("x"))) state.xKey = params.get("x");
      if (SPLIT_Y.some(a => a.key === params.get("sy"))) state.splitY = params.get("sy");
      if (SPLIT_ORDER.some(a => a.key === params.get("so"))) state.splitOrder = params.get("so");
      const ds = DATASETS.findIndex(x => x.key === params.get("ds"));
      if (ds >= 0) state.selected = ds;
    } catch (err) { /* keep the defaults */ }
    // The controls show the state read.
    for (const b of document.querySelectorAll(".hh-fchip")) b.setAttribute("aria-pressed", String(b.dataset.v === state[b.dataset.f]));
    document.getElementById("search").value = state.query;
    document.getElementById("bh").checked = state.bh;
    document.getElementById("yaxis").value = state.yKey;
    document.getElementById("xaxis").value = state.xKey;
    renderMarks();
  }

  // ---------- the list as CSV ----------
  // The list as it is filtered and sorted, one row per dataset, with both methods' names on every row.
  function tableCsv() {
    const head = ["method_a", "method_b", "dataset", "task", "train_rows", "features", "splits", "error_a", "error_b",
      "margin_pct", "rank_a", "rank_b", "delta_rank", "elo_pull", "a_lower_splits", "b_lower_splits", "tied_splits", "p",
      "q_bh", "verdict", "methods_sig_better_than_a", "train_s_a", "train_s_b", "predict_s_a", "predict_s_b"];
    const cell = v => {
      if (v == null || (typeof v === "number" && !isFinite(v))) return "";
      const text = typeof v === "number" ? String(+v.toPrecision(8)) : String(v);
      return /[",\n]/.test(text) ? `"${text.replace(/"/g, '""')}"` : text;
    };
    const lines = [head.join(",")];
    for (const d of visible) {
      const ds = DATASETS[d], x = PAIR[d];
      lines.push([METHODS[state.a].name, METHODS[state.b].name, ds.key, ds.task, trainRows(ds), ds.features, x.n,
        x.pa.e, x.pb.e, x.margin, x.pa.r, x.pb.r, x.dRank, ELO_PULL ? ELO_PULL.get(d) : null, x.wins, x.losses, x.ties,
        x.p, x.q, verdictOf(d), BETTER && BETTER[d] ? BETTER[d].methods.length : null, x.pa.t, x.pb.t, x.pa.it, x.pb.it]
        .map(cell).join(","));
    }
    return lines.join("\n") + "\n";
  }
  function flash(button, text) {
    const label = button.textContent;
    button.textContent = text;
    setTimeout(() => { button.textContent = label; }, 1600);
  }
  document.getElementById("btn-copy-csv").addEventListener("click", ev => {
    const button = ev.currentTarget;
    navigator.clipboard.writeText(tableCsv()).then(() => flash(button, "Copied"), () => flash(button, "Copy failed"));
  });
  document.getElementById("btn-csv").addEventListener("click", () => {
    const url = URL.createObjectURL(new Blob([tableCsv()], { type: "text/csv" }));
    downloadUrl(url, slugify(`${METHODS[state.a].name} vs ${METHODS[state.b].name}`) + ".csv");
    setTimeout(() => URL.revokeObjectURL(url), 5000);
  });

  // ---------- A against every method ----------
  // For each other method X, over the datasets in view: the paired Elo gap A - X (fewer resamples than
  // the tile, to keep the table quick), the error ratio, the datasets each leads and leads significantly
  // on, the split win rate and the mean ranks. Computed one method at a time while the section is open.
  const VS_RESAMPLES = 200;
  const VS_COLUMNS = [
    { key: "name", label: "Method X" },
    { key: "n", label: "Datasets", num: true },
    { key: "gap", label: "Elo gap A \u2212 X", num: true, hint: "Paired: A against the field without X, X against the field without A" },
    { key: "ci", label: "95% interval", num: true, sortable: false },
    { key: "above", label: "A ahead in", num: true, hint: "Share of the resamples in which A's Elo is above X's" },
    { key: "ratio", label: "Error ratio A / X", num: true },
    { key: "leadX", label: "X lower", num: true, hint: "Datasets where X has the lower mean error" },
    { key: "sigX", label: "X sig. better", num: true, hint: "Datasets where X's error is significantly lower than A's (as the Significant tile counts them)" },
    { key: "sigA", label: "A sig. better", num: true, hint: "Datasets where A's error is significantly lower than X's" },
    { key: "win", label: "A split wins", num: true, hint: "Share of the paired splits where A's error is lower, ties counting half" },
    { key: "rankX", label: "Rank X / A", num: true, hint: "Mean rank in the field over the datasets both ran on" },
  ];
  const vsState = { sort: "sigX", dir: -1, results: new Map(), ticket: 0 };
  function versus(x, datasets) {
    let n = 0, leadA = 0, leadX = 0, wins = 0, ties = 0, splits = 0, rankA = 0, rankX = 0;
    const logs = [], tested = [];
    for (const d of datasets) {
      const pa = key(d, state.a), px = key(d, x);
      if (!pa || !px) continue;
      n++;
      const diffs = [];
      let w = 0, l = 0, t = 0;
      for (let i = 0; i < pa.f.length; i++) {
        const u = pa.f[i], v = px.f[i];
        if (u == null || v == null) continue;
        if (u < v) w++; else if (u > v) l++; else t++;
        diffs.push(u - v);
      }
      wins += w; ties += t; splits += w + l + t;
      const lead = pa.e < px.e ? "a" : pa.e > px.e ? "x" : null;
      if (lead === "a") leadA++; else if (lead === "x") leadX++;
      if (pa.e > 0 && px.e > 0) logs.push(Math.log(pa.e / px.e));
      if (pa.r != null && px.r != null) { rankA += pa.r; rankX += px.r; }
      if (w + l + t >= MIN_SPLITS) tested.push({ p: signedRank(diffs).p, lead: lead });
    }
    const ps = state.bh ? bhAdjust(tested.map(r => r.p)) : tested.map(r => r.p);
    let sigA = 0, sigX = 0;
    tested.forEach((r, i) => { if (ps[i] < ALPHA && r.lead === "a") sigA++; if (ps[i] < ALPHA && r.lead === "x") sigX++; });
    const elo = eloGap(datasets, state.a, x, VS_RESAMPLES);
    return {
      n: n, leadA: leadA, leadX: leadX, sigA: sigA, sigX: sigX,
      ratio: logs.length ? Math.exp(logs.reduce((s, v) => s + v, 0) / logs.length) : null,
      win: splits ? (wins + 0.5 * ties) / splits : null,
      rankA: n ? rankA / n : null, rankX: n ? rankX / n : null,
      gap: elo ? elo.gap : null, lo: elo ? elo.lo : null, hi: elo ? elo.hi : null, above: elo ? elo.above : null,
    };
  }
  function scheduleVersus() {
    if (!document.getElementById("vsfield").open) return;
    const ticket = ++vsState.ticket;
    const datasets = [];
    for (let d = 0; d < nD; d++) if (passes(d, {})) datasets.push(d);
    const others = METHODS.map((m, i) => i).filter(i => i !== state.a && POINTS.some(p => p.m === i));
    const viewKey = `${state.a}|${state.bh}|${datasets.join(",")}`;
    if (vsState.viewKey !== viewKey) { vsState.results = new Map(); vsState.viewKey = viewKey; }
    renderVersus(others);
    let i = 0;
    const next = () => {
      if (ticket !== vsState.ticket) return;
      while (i < others.length && vsState.results.has(others[i])) i++;
      if (i >= others.length) return;
      vsState.results.set(others[i], versus(others[i], datasets));
      i++;
      renderVersus(others);
      setTimeout(next, 0);
    };
    setTimeout(next, 30);
  }
  function versusValue(x, r, sortKey) {
    if (sortKey === "name") return METHODS[x].name.toLowerCase();
    if (!r) return null;
    if (sortKey === "rankX") return r.rankX;
    return r[sortKey];
  }
  function renderVersus(others) {
    const head = document.getElementById("vs-head");
    head.innerHTML = "<tr>" + VS_COLUMNS.map(c =>
      `<th class="${c.num ? "num " : ""}${c.sortable === false ? "" : "sortable"}" data-k="${c.key}"` +
      `${c.hint ? ` title="${escapeHtml(c.hint)}"` : ""}${vsState.sort === c.key ? ` aria-sort="${vsState.dir > 0 ? "ascending" : "descending"}"` : ""}>` +
      `${c.label}</th>`).join("") + "</tr>";
    for (const th of head.querySelectorAll("th.sortable")) {
      th.addEventListener("click", () => {
        const k = th.dataset.k;
        if (vsState.sort === k) vsState.dir = -vsState.dir;
        else { vsState.sort = k; vsState.dir = ["sigX", "leadX", "above", "win", "n"].includes(k) ? -1 : 1; }
        renderVersus(others);
      });
    }
    const order = others.slice().sort((p, q) => {
      const vp = versusValue(p, vsState.results.get(p), vsState.sort), vq = versusValue(q, vsState.results.get(q), vsState.sort);
      if (vp == null && vq == null) return METHODS[p].name.localeCompare(METHODS[q].name);
      if (vp == null) return 1;
      if (vq == null) return -1;
      const c = typeof vp === "string" ? vp.localeCompare(vq) : vp - vq;
      return vsState.dir * c || METHODS[p].name.localeCompare(METHODS[q].name);
    });
    const body = document.getElementById("vs-body");
    body.innerHTML = order.map(x => {
      const r = vsState.results.get(x);
      const name = `<td>${escapeHtml(METHODS[x].name)}</td>`;
      if (!r) return `<tr data-m="${x}" class="clickable${x === state.b ? " is-b" : ""}">${name}<td class="num pending" colspan="${VS_COLUMNS.length - 1}">computing\u2026</td></tr>`;
      const excludes = r.lo != null && (r.lo > 0 || r.hi < 0);
      return `<tr data-m="${x}" class="clickable${x === state.b ? " is-b" : ""}" tabindex="0" title="Click to compare A with this method">${name}` +
        `<td class="num">${r.n}</td>` +
        `<td class="num${excludes ? " excludes0" : ""}">${r.gap == null ? "\u2014" : fmtSigned(r.gap, 1)}</td>` +
        `<td class="num">${r.lo == null ? "\u2014" : `[${fmtSigned(r.lo, 1)}, ${fmtSigned(r.hi, 1)}]`}</td>` +
        `<td class="num">${r.above == null ? "\u2014" : (r.above * 100).toFixed(0) + "%"}</td>` +
        `<td class="num">${r.ratio == null ? "\u2014" : r.ratio.toFixed(4)}</td>` +
        `<td class="num">${r.leadX} / ${r.n}</td>` +
        `<td class="num${r.sigX ? " excludes0" : ""}">${r.sigX}</td>` +
        `<td class="num">${r.sigA}</td>` +
        `<td class="num">${r.win == null ? "\u2014" : (r.win * 100).toFixed(1) + "%"}</td>` +
        `<td class="num">${r.rankX == null ? "\u2014" : `${r.rankX.toFixed(2)} / ${r.rankA.toFixed(2)}`}</td></tr>`;
    }).join("");
    for (const tr of body.querySelectorAll("tr.clickable")) {
      const act = () => {
        state.b = Number(tr.dataset.m);
        document.getElementById("pick-b").value = String(state.b);
        refresh();
      };
      tr.addEventListener("click", act);
      tr.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); act(); } });
    }
    postHeight();
  }
  document.getElementById("vsfield").addEventListener("toggle", scheduleVersus);

  // ---------- Elo gap by subset ----------
  // Every way of keeping the datasets the filters offer (and, for the column kinds, its complement),
  // over all datasets whatever the filters say, computed one row at a time after the page is up.
  function subsetRows() {
    const rows = [{ group: "", label: "All datasets", keep: () => true }];
    const kindsOf = d => DATASETS[d].kinds || [];
    const kinds = Object.keys(KIND_LABELS).filter(k => DATASETS.some(ds => (ds.kinds || []).includes(k)));
    for (const k of kinds) {
      rows.push({ group: "Columns", label: KIND_LABELS[k], keep: d => kindsOf(d).includes(k) });
      if (k !== "numerical") {
        rows.push({ group: "Columns", label: "Without " + KIND_LABELS[k].toLowerCase(), keep: d => !kindsOf(d).includes(k) });
      }
    }
    if (kinds.includes("text") && kinds.includes("high-cardinality")) {
      rows.push({
        group: "Columns", label: "Without text or high-card categorical",
        keep: d => !kindsOf(d).includes("text") && !kindsOf(d).includes("high-cardinality"),
      });
    }
    for (const t of tasksPresent) rows.push({ group: "Task", label: TASK_LABELS[t] || t, keep: d => DATASETS[d].task === t });
    const splits = Object.keys(SPLIT_LABELS).filter(k => DATASETS.some(ds => ds.split === k));
    if (splits.length > 1) {
      for (const k of splits) rows.push({ group: "Split", label: SPLIT_LABELS[k], keep: d => DATASETS[d].split === k });
    }
    for (const b of SIZE_BUCKETS.filter(b => SIZE_OF.includes(b.key))) {
      rows.push({ group: "Size", label: b.label, keep: d => SIZE_OF[d] === b.key });
    }
    return rows;
  }
  function eloGapOf(datasets) {
    const cacheKey = `${state.a}|${state.b}|${datasets.join(",")}`;
    if (!eloCache.has(cacheKey)) eloCache.set(cacheKey, eloGap(datasets));
    return eloCache.get(cacheKey);
  }
  // The geometric-mean error ratio A / B over the datasets both were measured on, as in the summary.
  function errorRatioOf(datasets) {
    const logs = datasets.map(d => PAIR[d] && PAIR[d].logRatio).filter(v => v != null);
    return logs.length ? Math.exp(logs.reduce((s, v) => s + v, 0) / logs.length) : null;
  }
  // The rows never change (they depend on the datasets alone); their results follow the pair.
  let SUBSET_ROWS = null;
  function allSubsetRows() { return SUBSET_ROWS || (SUBSET_ROWS = subsetRows()); }
  let subsetResults = [];
  let subsetTicket = 0;
  function scheduleSubsets() {
    const ticket = ++subsetTicket;
    const rows = allSubsetRows();
    const results = rows.map(() => undefined);
    subsetResults = results;
    renderSubsets(rows, results);
    let i = 0;
    const next = () => {
      if (ticket !== subsetTicket || i >= rows.length) return;
      const datasets = [];
      for (let d = 0; d < nD; d++) if (rows[i].keep(d)) datasets.push(d);
      results[i] = { n: datasets.length, elo: eloGapOf(datasets), ratio: errorRatioOf(datasets) };
      i++;
      renderSubsets(rows, results);
      setTimeout(next, 0);
    };
    setTimeout(next, 120);
  }
  function renderSubsets(rows, results) {
    const body = document.getElementById("subset-body");
    // One scale for every row's interval, so the rows compare at a glance; it always shows 0.
    let lo = 0, hi = 0;
    for (const r of results) if (r && r.elo) { lo = Math.min(lo, r.elo.lo); hi = Math.max(hi, r.elo.hi); }
    const span = (hi - lo) || 1;
    lo -= span * 0.04; hi += span * 0.04;
    const W = 240, H = 18, P = 6;
    const X = v => P + (v - lo) / (hi - lo) * (W - 2 * P);
    document.getElementById("forest-head").textContent = results.some(r => r && r.elo)
      ? `scale ${fmtSigned(lo, 0)} to ${fmtSigned(hi, 0)}` : "";
    let html = "", group = null;
    rows.forEach((row, i) => {
      if (row.group !== group) {
        group = row.group;
        if (group) html += `<tr class="group"><td colspan="7">${escapeHtml(group)}</td></tr>`;
      }
      const r = results[i];
      const cells = [`<td>${escapeHtml(row.label)}</td>`];
      if (r === undefined) {
        cells.push('<td class="num pending" colspan="6">computing\u2026</td>');
      } else if (!r.elo) {
        cells.push(`<td class="num">${r.n}</td><td class="num pending" colspan="5">too few datasets</td>`);
      } else {
        const e = r.elo, excludes = e.lo > 0 || e.hi < 0;
        const side = e.gap >= 0 ? "a" : "b";
        const forest = `<svg width="${W}" height="${H}" viewBox="0 0 ${W} ${H}" style="display:block">` +
          `<line x1="${X(0)}" y1="1" x2="${X(0)}" y2="${H - 1}" stroke="var(--muted)" stroke-width="1"/>` +
          `<line x1="${X(e.lo)}" y1="${H / 2}" x2="${X(e.hi)}" y2="${H / 2}" stroke="var(--ink)" stroke-width="2" stroke-linecap="round"/>` +
          `<circle cx="${X(e.gap)}" cy="${H / 2}" r="4.5" fill="var(--side-${side})" stroke="var(--card)" stroke-width="1.2"/></svg>`;
        cells.push(
          `<td class="num">${r.n}</td>`,
          `<td class="num${excludes ? " excludes0" : ""}">${fmtSigned(e.gap, 1)}</td>`,
          `<td class="num">[${fmtSigned(e.lo, 1)}, ${fmtSigned(e.hi, 1)}]</td>`,
          `<td class="forest">${forest}</td>`,
          `<td class="num">${(e.above * 100).toFixed(1)}%</td>`,
          `<td class="num">${r.ratio == null ? "\u2014" : r.ratio.toFixed(4)}</td>`,
        );
      }
      const mode = state.marks.get(row.label);
      const badge = `<span class="hh-markbadge">${mode === "in" ? "+" : mode === "out" ? "\u2212" : ""}</span>`;
      cells[0] = `<td>${badge}${escapeHtml(row.label)}</td>`;
      const hint = row.label === "All datasets" ? "Click to clear every mark"
        : mode === "in" ? "Kept: click to take these datasets out" : mode === "out" ? "Taken out: click to clear"
          : "Click to keep only these datasets";
      html += `<tr data-i="${i}" class="clickable${mode ? " mark-" + mode : ""}" tabindex="0" title="${hint}">${cells.join("")}</tr>`;
    });
    body.innerHTML = html;
    for (const tr of body.querySelectorAll("tr.clickable")) {
      const act = () => cycleMark(rows[Number(tr.dataset.i)]);
      tr.addEventListener("click", act);
      tr.addEventListener("keydown", ev => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); act(); } });
    }
    postHeight();
  }
  // None, kept, taken out, none again; "All datasets" clears every mark.
  function cycleMark(row) {
    if (row.label === "All datasets") state.marks.clear();
    else {
      const mode = state.marks.get(row.label);
      if (!mode) state.marks.set(row.label, "in");
      else if (mode === "in") state.marks.set(row.label, "out");
      else state.marks.delete(row.label);
    }
    renderMarks();
    renderSubsets(allSubsetRows(), subsetResults);
    refreshView();
  }
  function renderMarks() {
    const box = document.getElementById("marks");
    box.hidden = !state.marks.size;
    if (!state.marks.size) { box.textContent = ""; return; }
    box.innerHTML = '<span class="grouplabel">Subsets</span>';
    for (const [label, mode] of state.marks) {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "hh-mark" + (mode === "out" ? " out" : "");
      chip.textContent = `${mode === "in" ? "+" : "\u2212"} ${label} \u00d7`;
      chip.title = mode === "in" ? "Only these datasets are in view: click to remove the mark" : "These datasets are taken out of the view: click to remove the mark";
      chip.addEventListener("click", () => {
        state.marks.delete(label);
        renderMarks();
        renderSubsets(allSubsetRows(), subsetResults);
        refreshView();
      });
      box.appendChild(chip);
    }
    const clear = document.createElement("button");
    clear.type = "button";
    clear.className = "btn";
    clear.textContent = "Clear";
    clear.addEventListener("click", () => {
      state.marks.clear();
      renderMarks();
      renderSubsets(allSubsetRows(), subsetResults);
      refreshView();
    });
    box.appendChild(clear);
  }

  function refreshView() {
    scheduleBetter();
    scheduleVersus();
    computeView();
    summarize();
    buildLegend();
    renderList();
    drawChart();
    renderDetail();
  }
  function refresh() {
    computePair();
    writeHash();
    refreshView();
    scheduleSubsets();
  }

  document.addEventListener("keydown", ev => {
    const tag = (ev.target && ev.target.tagName) || "";
    if (tag === "INPUT" || tag === "SELECT" || tag === "TEXTAREA") {
      if (ev.key === "Escape") ev.target.blur();
      return;
    }
    if (ev.metaKey || ev.ctrlKey || ev.altKey) return;
    const moves = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1, PageDown: 10, PageUp: -10 };
    if (ev.key in moves) { ev.preventDefault(); step(moves[ev.key]); }
  });

  setUpPaperView(() => drawChart(), { openInPaper: false });
  setUpExport(() => [{ svg: document.getElementById("chart"), dx: 0 }],
    () => slugify(`${sideName("a")} vs ${sideName("b")}`));
  redrawOnResize(document.getElementById("chartbox"), drawChart);
  redrawOnResize(document.getElementById("detail"), drawSplits);

  readHash();
  buildPickers();
  buildHead();
  computePair();
  refreshView();
  scheduleSubsets();
  window.addEventListener("load", postHeight);
})();
</script>
</body>
</html>
"""
