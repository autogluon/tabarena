"""HTML/CSS/JS template for the self-contained composite leaderboard explorer.

One table: a row per method, a column per evaluation subset, the cells graded per column as
``composite_leaderboard.png`` grades them. Kept as a Python string constant so it ships with the
package without any build-system data-file configuration;
:func:`tabarena.plot.interactive.composite_explorer.build_composite_explorer_html` substitutes the
placeholders (see :func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

COMPOSITE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  body { padding: 8px 8px 12px; }
  .cl-search, .cl-pick {
    font: 500 12.5px/1.2 system-ui, sans-serif; color: var(--ink); background: var(--chip-bg);
    border: 1px solid var(--line); border-radius: 7px; padding: 5px 8px;
  }
  .cl-search { min-width: 180px; }
  .cl-search:focus-visible, .cl-pick:focus-visible { outline: 2px solid var(--accent); outline-offset: 2px; }
  .cl-fams { display: flex; flex-wrap: wrap; gap: 6px; margin: 0 0 10px; }
  .cl-chips { display: inline-flex; flex-wrap: wrap; gap: 4px 6px; }
  .cl-chip {
    display: inline-flex; align-items: center; gap: 4px; font: 600 11.5px/1 system-ui, sans-serif;
    color: var(--muted); border: 1px dashed var(--line); border-radius: 999px; padding: 2px 3px 2px 8px;
  }
  .cl-chip button {
    font: 700 13px/1 system-ui, sans-serif; color: var(--muted); background: none; border: 0;
    border-radius: 999px; padding: 1px 4px; cursor: pointer;
  }
  .cl-chip button:hover, .cl-chip button:focus-visible { color: var(--ink); background: var(--chip-bg); outline: none; }

  /* The table scrolls sideways under a sticky method column: twenty subsets do not fit a page. */
  .cl-wrap {
    overflow: auto; max-width: 100%; max-height: 78vh; border: 1px solid var(--line); border-radius: 10px;
    scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent;
  }
  table.cl { border-collapse: separate; border-spacing: 0; font-size: 12.5px; font-variant-numeric: tabular-nums; }
  table.cl th, table.cl td { padding: 4px 8px; white-space: nowrap; border-bottom: 1px solid var(--line); }
  table.cl thead th {
    position: sticky; top: 0; z-index: 3; background: var(--card); color: var(--muted);
    font-size: 10.5px; font-weight: 650; letter-spacing: 0.04em; text-align: right; cursor: pointer; user-select: none;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  table.cl thead th:hover { color: var(--accent); }
  table.cl thead th[aria-sort="ascending"]::after { content: " \25B2"; }
  table.cl thead th[aria-sort="descending"]::after { content: " \25BC"; }
  table.cl th.cl-pos, table.cl td.cl-pos { text-align: right; color: var(--muted); }
  table.cl .cl-stick { position: sticky; left: 0; z-index: 2; background: var(--card); }
  table.cl thead .cl-stick { z-index: 4; text-align: left; }
  table.cl .cl-stick2 { position: sticky; left: 34px; z-index: 2; background: var(--card); box-shadow: inset -1px 0 0 var(--line); }
  table.cl thead .cl-stick2 { z-index: 4; text-align: left; }
  table.cl th.cl-pos, table.cl td.cl-pos { min-width: 26px; }
  td.cl-name { font-weight: 620; cursor: pointer; max-width: 320px; overflow: hidden; text-overflow: ellipsis; }
  td.cl-name .dot { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: var(--fam); margin-right: 6px; }
  td.cl-metric { color: var(--muted); font-size: 11.5px; text-align: left; }
  td.cl-val { text-align: right; color: #14161a; }
  td.cl-val.top { font-weight: 750; }
  td.cl-val.dark { color: #fbfbf9; }
  tr.marked td.cl-name, tr.marked td.cl-pos, tr.marked td.cl-metric {
    background: color-mix(in srgb, var(--accent) 16%, var(--card));
  }
  tr.marked td.cl-name { box-shadow: inset 3px 0 0 var(--accent), inset -1px 0 0 var(--line); }
  .cl-empty { padding: 30px 8px; color: var(--muted); font-style: italic; }
  body.paper .controls, body.paper .cl-fams { display: none !important; }
  body.paper .cl-wrap { max-height: none; border: none; }
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, the table only">Paper view</button>
    <button class="btn" id="btn-csv" title="Download the table as shown (methods, order and metrics) as CSV">Download CSV</button>
  </div>
  <p class="explorer-title" id="title"></p>
  <div class="controls">
    <span class="grouplabel">Metric</span>
    <div class="btnrow" id="metric-btns"></div>
    <input class="cl-search" id="search" type="search" placeholder="Find a method" aria-label="Find a method">
    <span class="grouplabel">Hide</span>
    <select class="cl-pick" id="hidepick" aria-label="Hide a method"></select>
    <span class="cl-chips" id="hiddenlist"></span>
    <span class="hint">Click a column to sort by it, a method to mark it</span>
  </div>
  <!-- The paper view toggles an export bar; a table has no figure to export, so it stays empty. -->
  <div class="exportbar" id="exportbar" hidden></div>
  <div class="cl-fams" id="fams"></div>
  <div class="cl-wrap" id="wrap"><table class="cl" id="table"></table></div>
  <div class="tooltip"></div>

<script>
(function () {
  "use strict";
  const CONFIG = __CONFIG_JSON__;
  const POINTS = __POINTS_JSON__;

__BASE_JS__

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;
  const tip = makeTooltip(document.body);
  for (const p of POINTS) p.family = famOf(p.family);
  const SUBSETS = CONFIG.subsets, METRICS = CONFIG.metrics;
  const ORDER = POINTS.map((p, i) => i);

  const state = {
    metrics: [METRICS[0]],
    families: new Set(POINTS.map(p => p.family)),
    query: "",
    sort: null,          // { col, dir }: a subset index and its direction; null keeps the published order
    marked: new Set(),
    hidden: new Set(),   // models the reader took out
    hostExcluded: new Set(),
  };

  // ---------- which methods, in which order ----------
  function shown() {
    const q = state.query;
    const out = ORDER.filter(i => {
      const p = POINTS[i];
      if (!state.families.has(p.family)) return false;
      if (state.hidden.has(p.model) || state.hostExcluded.has(p.model)) return false;
      return !q || p.method.toLowerCase().includes(q);
    });
    if (state.sort) {
      const metric = state.metrics[0], vals = CONFIG.values[metric], { col, dir } = state.sort;
      out.sort((a, b) => {
        const va = vals[a][col], vb = vals[b][col];
        if (va == null && vb == null) return a - b;
        if (va == null) return 1;
        if (vb == null) return -1;
        return dir * (va - vb) || a - b;
      });
    }
    return out;
  }

  // ---------- colour, as the PNG grades a column: RdYlGn over the methods on screen ----------
  const RDYLGN = ["#a50026", "#d73027", "#f46d43", "#fdae61", "#fee08b", "#ffffbf", "#d9ef8b", "#a6d96a", "#66bd63", "#1a9850", "#006837"];
  function hexRgb(h) { return [1, 3, 5].map(i => parseInt(h.slice(i, i + 2), 16)); }
  function gradeColor(t) {
    const x = Math.max(0, Math.min(1, t)) * (RDYLGN.length - 1);
    const i = Math.min(RDYLGN.length - 2, Math.floor(x)), f = x - i;
    const a = hexRgb(RDYLGN[i]), b = hexRgb(RDYLGN[i + 1]);
    const c = a.map((v, k) => Math.round(v + (b[k] - v) * f));
    return { css: `rgb(${c[0]},${c[1]},${c[2]})`, dark: 0.2126 * c[0] + 0.7152 * c[1] + 0.0722 * c[2] < 110 };
  }
  // Per column and metric: the range over the methods shown, and each method's rank in it.
  function columnStats(rows, metric) {
    const vals = CONFIG.values[metric], better = CONFIG.higherBetter[metric] ? 1 : -1;
    return SUBSETS.map((_, j) => {
      const present = rows.map(i => vals[i][j]).filter(v => v != null);
      const lo = Math.min(...present), hi = Math.max(...present);
      const sorted = [...present].sort((a, b) => better * (b - a));
      return { lo: lo, hi: hi, better: better, rankOf: v => 1 + sorted.findIndex(u => u === v), n: present.length };
    });
  }
  function fmtValue(metric, v) {
    if (v == null) return "—";
    return /elo/i.test(metric) ? Math.round(v).toString() : v.toFixed(Math.abs(v) >= 10 ? 1 : 2);
  }

  // ---------- the table ----------
  const table = document.getElementById("table");
  function render() {
    const rows = shown();
    if (!rows.length) {
      table.innerHTML = '<tbody><tr><td class="cl-empty">No method matches the filters.</td></tr></tbody>';
      postHeight();
      return;
    }
    const both = state.metrics.length > 1;
    const stats = new Map(state.metrics.map(m => [m, columnStats(rows, m)]));
    let head = '<thead><tr><th class="cl-pos cl-stick">#</th><th class="cl-stick2" data-col="-1">Method</th>' +
      (both ? "<th></th>" : "");
    SUBSETS.forEach((s, j) => {
      const sort = state.sort && state.sort.col === j ? (state.sort.dir > 0 ? "ascending" : "descending") : null;
      head += `<th data-col="${j}"${sort ? ` aria-sort="${sort}"` : ""} title="Sort by ${escapeHtml(s)}">${escapeHtml(s)}</th>`;
    });
    head += "</tr></thead>";
    let body = "<tbody>";
    rows.forEach((i, pos) => {
      const p = POINTS[i];
      state.metrics.forEach((metric, k) => {
        const cls = state.marked.has(p.method) ? ' class="marked"' : "";
        body += `<tr data-i="${i}"${cls}>`;
        if (k === 0) {
          const span = both ? ` rowspan="${state.metrics.length}"` : "";
          body += `<td class="cl-pos cl-stick"${span}>${pos + 1}</td>` +
            `<td class="cl-name cl-stick2"${span} style="--fam:${famColor(p)}" title="${escapeHtml(p.method)} · ${escapeHtml(famLabel(p))}">` +
            `<span class="dot"></span>${escapeHtml(p.method + leakageMark(p.model))}</td>`;
        }
        if (both) body += `<td class="cl-metric">${escapeHtml(metric)}</td>`;
        const st = stats.get(metric), vals = CONFIG.values[metric];
        SUBSETS.forEach((_, j) => {
          const v = vals[i][j];
          if (v == null) { body += '<td class="cl-val">—</td>'; return; }
          const c = st[j], span = c.hi - c.lo;
          const t = span > 0 ? (c.better > 0 ? (v - c.lo) / span : (c.hi - v) / span) : 0.5;
          const g = gradeColor(t), rank = c.rankOf(v);
          body += `<td class="cl-val${rank <= 3 ? " top" : ""}${g.dark ? " dark" : ""}" data-j="${j}" data-m="${k}" ` +
            `style="background:${g.css}">${fmtValue(metric, v)}</td>`;
        });
        body += "</tr>";
      });
    });
    table.innerHTML = head + body + "</tbody>";
    for (const th of table.querySelectorAll("thead th[data-col]")) {
      th.addEventListener("click", () => {
        const col = Number(th.dataset.col);
        if (col < 0) { state.sort = null; render(); return; }
        const natural = CONFIG.higherBetter[state.metrics[0]] ? -1 : 1;
        state.sort = state.sort && state.sort.col === col ? { col: col, dir: -state.sort.dir } : { col: col, dir: natural };
        render();
      });
    }
    for (const td of table.querySelectorAll("td.cl-name")) {
      td.addEventListener("click", () => {
        const m = POINTS[Number(td.parentElement.dataset.i)].method;
        if (state.marked.has(m)) state.marked.delete(m); else state.marked.add(m);
        render();
      });
    }
    postHeight();
  }
  table.addEventListener("mousemove", ev => {
    const td = ev.target.closest ? ev.target.closest("td.cl-val[data-j]") : null;
    if (!td) { tip.hide(); return; }
    const i = Number(td.parentElement.dataset.i), j = Number(td.dataset.j);
    const p = POINTS[i], rows = shown();
    const lines = METRICS.map(metric => {
      const v = CONFIG.values[metric][i][j];
      const st = columnStats(rows, metric)[j];
      return `<div>${escapeHtml(metric)}: <b>${fmtValue(metric, v)}</b>` +
        (v != null ? ` <span class="t-var">(${st.rankOf(v)} of ${st.n} shown)</span>` : "") + "</div>";
    }).join("");
    tip.show(`<div class="t-name">${escapeHtml(p.method)}</div><div>${escapeHtml(SUBSETS[j])}</div>${lines}`, ev);
  });
  table.addEventListener("mouseleave", () => tip.hide());

  // ---------- controls ----------
  function escapeHtml(text) {
    return String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
  }
  const metricBox = document.getElementById("metric-btns");
  const metricChoices = [...METRICS.map(m => [m, [m]]), ...(METRICS.length > 1 ? [["Both", METRICS]] : [])];
  for (const [label, metrics] of metricChoices) {
    const b = document.createElement("button");
    b.className = "btn";
    b.type = "button";
    b.textContent = label;
    b.addEventListener("click", () => {
      state.metrics = metrics;
      // A sorted column stays sorted best first: the direction follows the metric it now sorts on.
      if (state.sort) state.sort.dir = CONFIG.higherBetter[metrics[0]] ? -1 : 1;
      syncMetrics();
      render();
    });
    b.dataset.label = label;
    metricBox.appendChild(b);
  }
  function syncMetrics() {
    const label = state.metrics.length > 1 ? "Both" : state.metrics[0];
    for (const b of metricBox.children) b.setAttribute("aria-pressed", String(b.dataset.label === label));
  }
  const famBox = document.getElementById("fams");
  for (const fam of FAM_ORDER) {
    const count = POINTS.filter(p => p.family === fam).length;
    if (!count) continue;
    const b = document.createElement("button");
    b.className = "famchip";
    b.type = "button";
    b.style.setProperty("--fam", FAM_VAR[fam]);
    b.innerHTML = famChipLabel(fam, count);
    b.addEventListener("click", () => {
      if (state.families.has(fam)) state.families.delete(fam); else state.families.add(fam);
      b.setAttribute("aria-pressed", String(state.families.has(fam)));
      render();
    });
    b.setAttribute("aria-pressed", "true");
    famBox.appendChild(b);
  }
  let searchTimer = null;
  document.getElementById("search").addEventListener("input", ev => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => { state.query = ev.target.value.trim().toLowerCase(); render(); }, 110);
  });
  // Hiding is by model, as the host's exclusions are: a model's variants leave together.
  function buildHide() {
    const pick = document.getElementById("hidepick");
    const models = [...new Set(POINTS.map(p => p.model))]
      .filter(m => !state.hidden.has(m) && !state.hostExcluded.has(m)).sort((a, b) => a.localeCompare(b));
    pick.innerHTML = '<option value="">Hide a method…</option>' +
      models.map(m => `<option value="${escapeHtml(m)}">${escapeHtml(m)}</option>`).join("");
    pick.onchange = ev => {
      if (!ev.target.value) return;
      state.hidden.add(ev.target.value);
      buildHide();
      render();
    };
    const box = document.getElementById("hiddenlist");
    const names = [...state.hidden].sort();
    box.innerHTML = names.map(m => `<span class="cl-chip">${escapeHtml(m)}<button type="button" aria-label="Show ${escapeHtml(m)} again" ` +
      `title="Show ${escapeHtml(m)} again">×</button></span>`).join("");
    box.querySelectorAll("button").forEach((b, k) => b.addEventListener("click", () => {
      state.hidden.delete(names[k]);
      buildHide();
      render();
    }));
  }
  onExcludeMessage(excluded => { state.hostExcluded = excluded; buildHide(); render(); });

  document.getElementById("btn-csv").addEventListener("click", () => {
    const rows = shown();
    const cell = v => (v == null ? "" : /[",\n]/.test(String(v)) ? `"${String(v).replace(/"/g, '""')}"` : String(v));
    const lines = [["method", "metric", ...SUBSETS].map(cell).join(",")];
    for (const i of rows) {
      for (const metric of state.metrics) {
        lines.push([POINTS[i].method, metric, ...CONFIG.values[metric][i]].map(cell).join(","));
      }
    }
    const url = URL.createObjectURL(new Blob([lines.join("\n") + "\n"], { type: "text/csv" }));
    downloadUrl(url, slugify(CONFIG.title || document.title) + ".csv");
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  });

  setUpPaperView(render, { openInPaper: false });
  syncMetrics();
  buildHide();
  render();
  window.addEventListener("load", postHeight);
})();
</script>
</body>
</html>
"""
