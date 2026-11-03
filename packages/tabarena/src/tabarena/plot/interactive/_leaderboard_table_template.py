"""HTML/CSS/JS template for the interactive full leaderboard table.

The interactive twin of the website's leaderboard table: every column sortable,
per-column heat shading, family toggles, a methods menu, tuning-variant toggles, a
columns menu, search, and CSV export of exactly what is on screen.

It exists here rather than in the leaderboard Space so that it shares the
explorers' tokens and components — the family and variant colors, ``.chip`` /
``.famchip`` / ``.imp-mark``, paper view, and the iframe height protocol — by
reusing them instead of reimplementing them in the Space's Gradio CSS, where they
drifted. See :mod:`tabarena.plot.interactive._explorer_shared`.

Kept as a Python string constant (rather than a package-data file) so it ships
with the package without any build-system data-file configuration.
:func:`tabarena.plot.interactive.leaderboard_table.build_leaderboard_table_html`
substitutes the placeholders (see
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

LEADERBOARD_TABLE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  /* --- The table ------------------------------------------------------------
     A scroll box with a sticky header, so the column names stay put through 80
     rows. Numbers are tabular so digits line up down a column. */
  .lbt-scroll {
    overflow: auto; max-height: 720px; margin-top: 4px;
    border: 1px solid var(--line); border-radius: 10px;
    scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent;
  }
  .lbt-scroll::-webkit-scrollbar { width: 11px; height: 11px; }
  .lbt-scroll::-webkit-scrollbar-thumb {
    background: var(--pt-muted); border-radius: 8px;
    border: 3px solid transparent; background-clip: content-box;
  }
  table.lbt { border-collapse: collapse; width: 100%; font-variant-numeric: tabular-nums; }
  table.lbt th, table.lbt td { padding: 5px 9px; text-align: center; border: 1px solid var(--line); }
  table.lbt thead th {
    position: sticky; top: 0; z-index: 3; background: var(--card); font-size: 12px;
    font-weight: 650; white-space: nowrap; cursor: pointer; user-select: none;
    box-shadow: inset 0 -1px 0 var(--line);
  }
  /* The header is both the sort control and the definition, so it gets the
     pointer of the former and the dotted underline of the latter. */
  table.lbt thead th .lbl { text-decoration: underline; text-decoration-style: dotted;
    text-decoration-color: var(--muted); text-underline-offset: 3px; }
  table.lbt thead th:hover { color: var(--accent); }
  table.lbt thead th::after { content: "\2195"; font-size: 0.72em; opacity: 0.3; margin-left: 5px; }
  table.lbt thead th[aria-sort="ascending"]::after { content: "\25B2"; opacity: 0.85; }
  table.lbt thead th[aria-sort="descending"]::after { content: "\25BC"; opacity: 0.85; }
  table.lbt td.name { text-align: left; white-space: nowrap; }
  table.lbt td.pos { color: var(--muted); font-variant-numeric: tabular-nums; }
  table.lbt td.type { white-space: nowrap; width: 1%; }
  table.lbt td.num { font-weight: 600; white-space: nowrap; }
  table.lbt td.na { color: var(--pt-muted); }
  table.lbt tbody tr:hover td { filter: brightness(1.12); }
  /* A method the host page follows (its "Follow" picker) and one under the pointer in another
     chart of the page. */
  table.lbt tbody tr.followed td:first-child { box-shadow: inset 4px 0 0 var(--accent); }
  table.lbt tbody tr.followed td.name { background: color-mix(in srgb, var(--accent) 16%, transparent); }
  table.lbt tbody tr.hovered td { box-shadow: inset 0 2px 0 var(--ink), inset 0 -2px 0 var(--ink); }
  body.embedded table.lbt tbody tr { cursor: pointer; }
  .ci { opacity: 0.5; font-weight: 400; font-size: 0.82em; }
  .pill { padding: 1px 7px; border-radius: 999px; font-size: 0.95em; white-space: nowrap; }
  .verified { font-size: 0.85em; }
  /* Caveat chips on a system's name: what a reader has to weigh before comparing it to a
     model. Muted so they read as an annotation, not as part of the name. */
  .tag-chip {
    margin-left: 4px; padding: 0 5px; border-radius: 999px;
    font-size: 0.78em; font-weight: 500; white-space: nowrap;
    color: var(--muted); background: var(--chip-bg); border: 1px solid var(--line);
  }
  .link-icon { font-size: 0.78em; opacity: 0.65; margin-left: 2px; }
  /* Grey rather than a dimmed inherit: dimming leaves the tag tinted with the
     family colour, which the model name already carries, and made this table
     disagree with the cross-subset table on the app page. */
  .variant-tag { color: var(--muted); font-weight: 400; font-size: 0.9em; }
  td.name .imp-mark { color: var(--muted); font-weight: 700; margin-left: 3px; }
  td.num .fit-mark { font-weight: 700; margin-left: 2px; cursor: help; }
  td.name a { text-decoration: underline; text-decoration-style: dotted; text-underline-offset: 3px; }
  td.name a:hover { text-decoration-style: solid; }

  .lbt-search {
    font: 500 12.5px/1 system-ui, sans-serif; color: var(--ink);
    background: var(--chip-bg); border: 1px solid var(--line); border-radius: 7px;
    padding: 6px 9px; width: 12em;
  }
  .lbt-cap { font-size: 12.5px; color: var(--muted); margin: 7px 0 0; }
  .lbt-empty { padding: 34px 0; text-align: center; color: var(--muted); font-size: 13px; }
  /* One toolbar above the rows: search, the family toggles, the variant toggles and two menus.
     The per-method chips live in the Methods menu, not in a wall above the table. */
  .lbt-tools { gap: 8px 10px; }
  .lbt-fams { display: flex; flex-wrap: wrap; gap: 6px; }
  .lbt-gap { flex: 1 1 auto; }
  .lbt-pop { position: relative; }
  .lbt-pop > summary { list-style: none; }
  .lbt-pop > summary::-webkit-details-marker { display: none; }
  .lbt-pop > summary::after { content: " \25BE"; color: var(--muted); }
  .lbt-pop[open] > summary { border-color: var(--muted); }
  /* Opens rightwards: the menus sit at the start of the toolbar's second line, where the frame's left
     edge would cut a menu that opened the other way. */
  .lbt-menu {
    position: absolute; top: calc(100% + 6px); left: 0; z-index: 20;
    width: max-content; max-width: min(640px, 92vw); max-height: 440px; overflow-y: auto;
    padding: 10px 12px; border: 1px solid var(--line); border-radius: 10px;
    background: var(--card); box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
  }
  .lbt-menu .chips { gap: 10px; }
  .lbt-menu .famname { font-size: 12px; font-weight: 650; color: var(--muted); }
  .lbt-menu-head { display: flex; gap: 6px; margin-bottom: 9px; }
  #col-btns { display: flex; flex-direction: column; align-items: stretch; gap: 6px; }
  /* A method the latest releases added (the host's `tabarena-explorer-new`). */
  .new-tag {
    margin-left: 5px; padding: 0 6px; border-radius: 999px; vertical-align: 1px;
    font-size: 0.7em; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase;
    color: var(--accent); border: 1px solid color-mix(in srgb, var(--accent) 55%, transparent);
  }

  /* The data export, in the same green the website's figure-export controls use,
     so "this downloads something" reads the same across the site. */
  .btn.export {
    color: #063; background: #7ee0b8; border-color: #7ee0b8; font-weight: 700;
  }
  .btn.export:hover { background: #a6efcd; border-color: #a6efcd; }
  /* Paper view keeps the table: here the table *is* the figure. */
  body.paper .lbt-scroll { max-height: none; }
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, table only, for slides and papers">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>
  <div class="controls lbt-tools">
    <input class="lbt-search" id="search" type="search" placeholder="Find a method" aria-label="Find a method">
    <div class="lbt-fams" id="fam-btns"></div>
    <span class="lbt-gap"></span>
    <div class="btnrow" id="variant-btns"></div>
    <button class="btn" id="btn-imputed" title="Models whose score is partly imputed, marked &#8225;"></button>
    <details class="lbt-pop" id="methods-pop">
      <summary class="btn" id="methods-sum">Methods</summary>
      <div class="lbt-menu">
        <div class="lbt-menu-head">
          <button class="btn" id="btn-all-methods">All</button>
          <button class="btn" id="btn-no-methods">None</button>
        </div>
        <div class="chips" id="chips"></div>
      </div>
    </details>
    <details class="lbt-pop" id="columns-pop">
      <summary class="btn">Columns</summary>
      <div class="lbt-menu" id="col-btns"></div>
    </details>
    <button class="btn export" id="btn-csv" title="Download the rows and columns shown, in the current sort order">Download CSV</button>
  </div>
  <!-- The shared paper-view helper reveals this bar; a table has no figure to
       export as SVG/PNG/PDF, and its CSV button belongs beside the filters that
       shape the export, so it stays empty here. -->
  <div class="exportbar" id="exportbar" hidden></div>
  <div class="lbt-scroll" id="tblwrap"></div>
  <p class="lbt-cap" id="caption"></p>

<script>
(function () {
  "use strict";
  const CONFIG = __CONFIG_JSON__;
  const POINTS = __POINTS_JSON__;

__BASE_JS__

  const COLUMNS = CONFIG.columns;
  const RANK_KEY = CONFIG.rankKey;
  const colByKey = new Map(COLUMNS.map(c => [c.key, c]));
  const VARIANTS = CONFIG.variants.filter(v => POINTS.some(p => p.variant === v));

  // How a variant is written out. The short "Tuned + Ens." form is the internal key
  // (the charts' colour and marker tables are keyed by it), but nothing on the page
  // should show the abbreviation: the row tag uses the spelling the data and the
  // app's cross-subset table use, and the toggle spells it out in full.
  const VARIANT_TEXT = {
    "Default": "default",
    "Tuned": "tuned",
    "Tuned + Ens.": "tuned + ensembled",
  };
  const VARIANT_BTN = { "Tuned + Ens.": "Tuned + Ensembled" };
  const variantText = v => VARIANT_TEXT[v] || v;
  const HAS_IMPUTED = POINTS.some(p => p.imputed);
  const HAS_FIT_AT_PREDICT = POINTS.some(p => p.fits_at_predict);
  const FIT_AT_PREDICT_HINT = "This method fits inside its prediction call, so its train time is the time of " +
    "that call, from training data to predictions. Its predict time holds the fit too.";

  const state = {
    methods: new Set(POINTS.map(p => p.method)),
    variants: new Set(VARIANTS),
    // Methods the host took out of the page (see `onExcludeMessage`): no row, no chip, and
    // left out of the CSV export, which is built from `visibleRows`.
    excluded: new Set(),
    columns: new Set(COLUMNS.filter(c => c.on !== false).map(c => c.key)),
    imputed: true,
    search: "",
    sortKey: null,
    sortAsc: false,
  };
  // Methods the host asks every chart to follow (`onFollowMessage`), and the method under the
  // pointer in another chart of the page (`onHoverMessage`).
  let followed = new Set();
  let hovered = null;
  // Methods the latest releases added, each with its hover text (the host's
  // `tabarena-explorer-new`), badged "new" beside the name.
  let fresh = new Map();

  const tblwrap = document.getElementById("tblwrap");
  const chipsBox = document.getElementById("chips");
  const famChips = new Map();
  const chipByMethod = new Map();

  // ---------- heat shading ----------
  // Green (best) through olive to red (worst), matching the website's
  // cross-subset overview so a reader moving between them reads one scale. On a
  // light surface each stop is lifted to the pastel the website's light theme uses,
  // under dark digits.
  const RAMP = [[0, [28, 120, 62]], [0.5, [138, 122, 36]], [1, [160, 58, 58]]];
  const LIGHT_RAMP = [[0, [135, 201, 160]], [0.5, [224, 204, 123]], [1, [223, 144, 144]]];
  function lightSurface() {
    const theme = document.documentElement.getAttribute("data-theme");
    return theme ? theme === "light" : !window.matchMedia("(prefers-color-scheme: dark)").matches;
  }
  function heatColor(frac) {
    const ramp = lightSurface() ? LIGHT_RAMP : RAMP;
    const f = Math.max(0, Math.min(1, frac));
    for (let i = 0; i < ramp.length - 1; i++) {
      const [f0, c0] = ramp[i], [f1, c1] = ramp[i + 1];
      if (f <= f1) {
        const t = f1 === f0 ? 0 : (f - f0) / (f1 - f0);
        const mix = c0.map((v, j) => Math.round(v + t * (c1[j] - v)));
        return "rgb(" + mix.join(",") + ")";
      }
    }
    return "rgb(" + ramp[ramp.length - 1][1].join(",") + ")";
  }
  // Runtimes span orders of magnitude; shading them linearly paints every model
  // the same green and only the slowest one red, so those normalize in log space.
  function scaleOf(col, v) { return col.logScale ? Math.log10(v) : v; }
  function bounds(rows, col) {
    const values = [];
    for (const p of rows) {
      const v = p[col.key];
      if (v == null || !isFinite(v)) continue;
      if (col.logScale && !(v > 0)) continue;
      values.push(scaleOf(col, v));
    }
    if (values.length < 2) return null;
    const lo = Math.min(...values), hi = Math.max(...values);
    return hi > lo ? [lo, hi] : null;
  }

  // ---------- cells ----------
  function nameCell(p) {
    const ink = famInk(p) || "var(--muted)";
    let inner = escapeHtml(p.method);
    if (p.variant) inner += ' <span class="variant-tag">(' + escapeHtml(variantText(p.variant)) + ")</span>";
    if (p.verified) inner += ' <span class="verified" title="Verified implementation">&#10004;&#65039;</span>';
    if (fresh.has(p.method)) inner += '<span class="new-tag" title="' + escapeHtml(fresh.get(p.method)) + '">new</span>';
    const leaked = leakageOf(p.method);
    if (leaked) {
      inner += ' <span class="leak-tag" title="Leakage warning: test data leaked into its training, proven on ' +
        escapeHtml(leaked.join(", ")) + ', possible on any other dataset">&#9888;&#65038; Leakage</span>';
    }
    for (const tag of (p.tags || [])) {
      const spec = (CONFIG.tagSpecs || {})[tag];
      if (!spec) continue;
      inner += ' <span class="tag-chip" title="' + escapeHtml(spec.hint) + '">' +
        spec.emoji + " " + escapeHtml(spec.label) + "</span>";
    }
    if (p.imputed) {
      const pct = isFinite(p.imputed_pct) ? fmtNum(p.imputed_pct, 0) + "% " : "";
      inner += ' <span class="imp-mark" title="' + pct + 'imputed">&#8225;</span>';
    }
    const body = p.url
      ? '<a href="' + escapeHtml(p.url) + '" target="_blank" rel="noopener" style="color:' + ink +
        ';font-weight:600;">' + inner + '<span class="link-icon">&#8599;</span></a>'
      : '<span style="color:' + ink + ';font-weight:600;">' + inner + "</span>";
    return '<td class="name" data-export="' + escapeHtml(plainName(p)) + '">' + body + "</td>";
  }
  function plainName(p) {
    return p.method + (p.variant ? " (" + p.variant + ")" : "");
  }
  function escapeHtml(text) {
    return String(text == null ? "" : text).replace(/[&<>"']/g, c =>
      ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[c]);
  }

  // ---------- rows ----------
  function visibleRows() {
    const term = state.search.trim().toLowerCase();
    return POINTS.filter(p => {
      if (!(state.methods.has(p.method) || followed.has(p.method)) || state.excluded.has(p.method)) return false;
      if (p.variant && !state.variants.has(p.variant)) return false;
      if (!state.imputed && p.imputed) return false;
      if (term && !(p.method + " " + (p.variant || "") + " " + p.family).toLowerCase().includes(term)) return false;
      return true;
    });
  }
  function sortRows(rows) {
    const out = rows.slice();
    if (!state.sortKey) return out.sort((a, b) => a.position - b.position);
    const col = colByKey.get(state.sortKey);
    const sign = state.sortAsc ? 1 : -1;
    const keyOf = p => {
      const v = p[state.sortKey];
      if (col && col.text) return String(v == null ? "" : v).toLowerCase();
      return v == null || !isFinite(v) ? null : v;
    };
    return out.sort((a, b) => {
      const ka = keyOf(a), kb = keyOf(b);
      if (ka === null || kb === null) return ka === kb ? 0 : ka === null ? 1 : -1;
      return ka < kb ? -sign : ka > kb ? sign : 0;
    });
  }

  function activeColumns() {
    return COLUMNS.filter(c => c.always || state.columns.has(c.key));
  }

  function render() {
    const rows = sortRows(visibleRows());
    const cols = activeColumns();
    if (!rows.length) {
      tblwrap.innerHTML = '<p class="lbt-empty">Nothing selected — turn a family or variant back on.</p>';
      document.getElementById("caption").textContent = "";
      postHeight();
      return;
    }
    // No medals here: this table is the full ranking and already carries the
    // position column, so podium marks only added a second, competing ordering.
    const heat = new Map();
    for (const col of cols) if (col.heatmap) heat.set(col.key, bounds(rows, col));

    let html = '<table class="lbt"><thead><tr>';
    for (const col of cols) {
      const sorted = state.sortKey === col.key
        ? ' aria-sort="' + (state.sortAsc ? "ascending" : "descending") + '"' : "";
      html += "<th" + sorted + ' data-key="' + col.key + '" title="' + escapeHtml(col.hint || col.label) +
        '"><span class="lbl">' + escapeHtml(col.label) + "</span>" +
        (col.key === "elo" ? ' <span class="ci">(95% CI)</span>' : "") + "</th>";
    }
    html += "</tr></thead><tbody>";
    for (const p of rows) {
      const marks = (followed.has(p.method) ? " followed" : "") + (p.method === hovered ? " hovered" : "");
      html += '<tr data-m="' + escapeHtml(p.method) + '"' + (marks ? ' class="' + marks.trim() + '"' : "") + ">";
      for (const col of cols) {
        if (col.key === "position") {
          html += '<td class="pos num" data-sort="' + p.position + '">' + (p.position + 1) + "</td>";
        } else if (col.key === "family") {
          const mark = famColor(p) || "var(--fam-baseline)";
          html += '<td class="type" data-sort="' + escapeHtml(p.family) + '" data-export="' + escapeHtml(p.family) +
            '"><span class="pill" style="background:color-mix(in srgb, ' + mark +
            ' 13%, transparent);color:' + (famInk(p) || "var(--muted)") +
            ";border:1px solid color-mix(in srgb, " + mark + ' 40%, transparent);">' +
            escapeHtml(p.family_symbol || "") + "</span></td>";
        } else if (col.key === "model") {
          html += nameCell(p);
        } else {
          html += valueCell(p, col, heat.get(col.key));
        }
      }
      html += "</tr>";
    }
    tblwrap.innerHTML = html + "</tbody></table>";
    for (const th of tblwrap.querySelectorAll("th")) {
      th.addEventListener("click", () => sortBy(th.dataset.key));
    }
    document.getElementById("caption").textContent =
      rows.length + (rows.length === 1 ? " row" : " rows") +
      " · click a column header to sort, hover one for what it means" +
      (heat.size ? " · green is better, red is worse, per column" : "") +
      (HAS_IMPUTED ? " · ‡ marks a partly imputed score" : "") +
      (HAS_FIT_AT_PREDICT ? " · † train time includes the prediction call" : "");
    postHeight();
  }

  function valueCell(p, col, span) {
    const v = p[col.key];
    if (v == null || (typeof v === "number" && !isFinite(v))) return '<td class="na">&ndash;</td>';
    if (col.text) {
      return '<td data-sort="' + escapeHtml(v) + '" data-export="' + escapeHtml(v) + '">' + escapeHtml(v) + "</td>";
    }
    let style = "";
    if (span) {
      let frac = (scaleOf(col, v) - span[0]) / (span[1] - span[0]);
      if (!col.lowerBetter) frac = 1 - frac;
      style = ' style="background:' + heatColor(frac) + ";color:" + (lightSurface() ? "#14161a" : "#f7f7f7") + ';"';
    }
    let text = fmtNum(v, col.decimals) + (col.suffix || "");
    if (col.key === "train_time" && p.fits_at_predict) {
      text += '<span class="fit-mark" title="' + FIT_AT_PREDICT_HINT + '">&dagger;</span>';
    }
    let ci = "";
    if (col.key === "elo" && p.elo_ci) {
      ci = ' <span class="ci">(' + escapeHtml(p.elo_ci) + ")</span>";
    }
    return '<td class="num" data-sort="' + v + '"' + (ci ? ' data-ci="' + escapeHtml(p.elo_ci) + '"' : "") +
      style + ">" + text + ci + "</td>";
  }

  function sortBy(key) {
    if (!key) return;
    if (state.sortKey === key) state.sortAsc = !state.sortAsc;
    else { state.sortKey = key; state.sortAsc = !!(colByKey.get(key) || {}).lowerBetter; }
    render();
  }

  // ---------- chips (families and their methods) ----------
  function familyMembers(fam) {
    return [...new Set(POINTS.filter(p => p.family === fam && !state.excluded.has(p.method)).map(p => p.method))];
  }
  function methodRank(name) {
    const col = colByKey.get(RANK_KEY);
    const values = POINTS.filter(p => p.method === name && isFinite(p[RANK_KEY])).map(p => p[RANK_KEY]);
    if (!values.length) return Infinity;
    return col.lowerBetter ? Math.min(...values) : -Math.max(...values);
  }
  // The family toggles sit in the toolbar; each method's chip sits in the Methods menu, under
  // its family's name.
  function buildChips() {
    const famBox = document.getElementById("fam-btns");
    for (const fam of FAM_ORDER) {
      const members = familyMembers(fam);
      if (!members.length) continue;
      members.sort((a, b) => methodRank(a) - methodRank(b));
      const famBtn = document.createElement("button");
      famBtn.className = "famchip";
      famBtn.style.setProperty("--fam", FAM_VAR[fam]);
      famBtn.innerHTML = famChipLabel(fam, members.length);
      famBtn.title = "Show or hide all " + members.length + " " + fam + " methods";
      famBtn.addEventListener("click", () => toggleFamily(fam));
      famBox.appendChild(famBtn);
      famChips.set(fam, famBtn);
      const row = document.createElement("div");
      row.className = "chiprow";
      const name = document.createElement("span");
      name.className = "famname";
      name.textContent = fam;
      row.appendChild(name);
      const set = document.createElement("div");
      set.className = "chipset";
      for (const name of members) {
        const imputed = POINTS.some(p => p.method === name && p.imputed);
        const b = document.createElement("button");
        b.className = "chip";
        b.style.setProperty("--fam", famColor(POINTS.find(p => p.method === name)));
        b.innerHTML = '<span class="dot"></span><span>' + escapeHtml(name) + "</span>" +
          (imputed ? '<span class="imp-mark">&#8225;</span>' : "");
        b.title = name + (imputed ? " (partially imputed)" : "");
        b.addEventListener("click", () => toggleMethod(name));
        set.appendChild(b);
        chipByMethod.set(name, b);
      }
      row.appendChild(set);
      chipsBox.appendChild(row);
    }
  }
  function syncChips() {
    for (const [name, b] of chipByMethod) {
      b.setAttribute("aria-pressed", String(state.methods.has(name)));
      b.hidden = state.excluded.has(name);
    }
    for (const [fam, b] of famChips) {
      const members = familyMembers(fam);
      b.hidden = !members.length;
      b.setAttribute("aria-pressed", String(members.every(m => state.methods.has(m))));
      b.innerHTML = famChipLabel(fam, members.length);
    }
    // The menu's button says how many of the methods are shown, when not all of them are.
    const all = [...chipByMethod.keys()].filter(m => !state.excluded.has(m));
    const shown = all.filter(m => state.methods.has(m)).length;
    document.getElementById("methods-sum").textContent =
      shown === all.length ? "Methods" : "Methods " + shown + "/" + all.length;
  }
  function setMethods(names) {
    state.methods = new Set(names);
    syncChips();
    render();
  }
  onExcludeMessage(excluded => {
    state.excluded = excluded;
    syncChips();
    render();
  });
  function toggleMethod(name) {
    if (state.methods.has(name)) state.methods.delete(name); else state.methods.add(name);
    syncChips();
    render();
  }
  function toggleFamily(fam) {
    const members = familyMembers(fam);
    const allOn = members.every(m => state.methods.has(m));
    for (const m of members) { if (allOn) state.methods.delete(m); else state.methods.add(m); }
    syncChips();
    render();
  }

  // ---------- variant / imputed / column toggles ----------
  const variantBtns = new Map();
  function buildVariantBtns() {
    const box = document.getElementById("variant-btns");
    for (const v of VARIANTS) {
      const b = document.createElement("button");
      b.className = "btn toggle";
      b.innerHTML = '<span class="swatch"></span>' + escapeHtml(VARIANT_BTN[v] || v);
      b.style.setProperty("--fam", VARIANT_VAR[v] || "var(--accent)");
      b.title = "Show or hide the " + variantText(v) + " results";
      b.addEventListener("click", () => {
        if (state.variants.has(v)) state.variants.delete(v); else state.variants.add(v);
        syncVariantBtns();
        render();
      });
      box.appendChild(b);
      variantBtns.set(v, b);
    }
  }
  function syncVariantBtns() {
    for (const [v, b] of variantBtns) b.setAttribute("aria-pressed", String(state.variants.has(v)));
    const imp = document.getElementById("btn-imputed");
    imp.hidden = !HAS_IMPUTED;
    imp.className = "btn toggle";
    imp.innerHTML = '<span class="swatch"></span>&#8225; Imputed';
    imp.setAttribute("aria-pressed", String(state.imputed));
    imp.style.setProperty("--fam", "var(--muted)");
  }
  function buildColumnBtns() {
    const box = document.getElementById("col-btns");
    for (const col of COLUMNS) {
      if (col.always) continue;
      const b = document.createElement("button");
      b.className = "btn toggle";
      // The menu has the room for the full name.
      b.textContent = col.label;
      b.title = col.hint || col.label;
      b.style.setProperty("--fam", "var(--accent)");
      b.addEventListener("click", () => {
        if (state.columns.has(col.key)) state.columns.delete(col.key); else state.columns.add(col.key);
        b.setAttribute("aria-pressed", String(state.columns.has(col.key)));
        render();
      });
      b.setAttribute("aria-pressed", String(state.columns.has(col.key)));
      box.appendChild(b);
    }
  }

  // ---------- CSV ----------
  // Exports what the reader is looking at: the rows the filters left, the
  // columns they picked, in the order they sorted.
  function downloadCsv() {
    const cols = activeColumns();
    const rows = sortRows(visibleRows());
    const quote = v => '"' + String(v == null ? "" : v).replace(/"/g, '""') + '"';
    const head = [];
    for (const col of cols) {
      head.push(col.label);
      if (col.key === "elo" && rows.some(p => p.elo_ci)) head.push("Elo 95% CI");
    }
    const lines = [head.map(quote).join(",")];
    for (const p of rows) {
      const values = [];
      for (const col of cols) {
        if (col.key === "model") values.push(plainName(p));
        else if (col.key === "position") values.push(p.position + 1);
        else values.push(p[col.key]);
        if (col.key === "elo" && rows.some(q => q.elo_ci)) values.push(p.elo_ci || "");
      }
      lines.push(values.map(quote).join(","));
    }
    const blob = new Blob([lines.join("\n") + "\n"], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    downloadUrl(url, slugify(CONFIG.title || document.title || "leaderboard") + ".csv");
    URL.revokeObjectURL(url);
  }

  // ---------- boot ----------
  if (CONFIG.title) document.getElementById("title").textContent = CONFIG.title;
  document.getElementById("search").addEventListener("input", ev => {
    state.search = ev.target.value;
    render();
  });
  document.getElementById("btn-imputed").addEventListener("click", () => {
    state.imputed = !state.imputed;
    syncVariantBtns();
    render();
  });
  document.getElementById("btn-csv").addEventListener("click", downloadCsv);
  document.getElementById("btn-all-methods").addEventListener("click", () => setMethods(POINTS.map(p => p.method)));
  document.getElementById("btn-no-methods").addEventListener("click", () => setMethods([]));
  // A menu closes on a click anywhere outside it, and on Escape.
  const menus = [...document.querySelectorAll(".lbt-pop")];
  document.addEventListener("click", ev => {
    for (const menu of menus) if (menu.open && !menu.contains(ev.target)) menu.open = false;
  });
  document.addEventListener("keydown", ev => {
    if (ev.key === "Escape") for (const menu of menus) menu.open = false;
  });
  // The host names the methods its latest releases added, with the hover text for each; a repeat
  // of the same set (sent with every height report) is ignored.
  let freshKey = "";
  window.addEventListener("message", ev => {
    const d = ev.data;
    if (!d || d.type !== "tabarena-explorer-new" || !d.methods || typeof d.methods !== "object") return;
    const next = new Map(Object.entries(d.methods).filter(([, hint]) => typeof hint === "string"));
    const key = [...next.keys()].sort().join("\u0000");
    if (key === freshKey) return;
    freshKey = key;
    fresh = next;
    render();
  });
  // Embedded, the host page puts its own CSV button in the panel header (next to
  // the title) and asks for the download over postMessage, the same way it drives
  // the figure exports. Standalone, this page needs its own button.
  if (window.parent !== window) {
    document.getElementById("btn-csv").hidden = true;
    window.addEventListener("message", ev => {
      if (ev.data && ev.data.type === "tabarena-leaderboard-csv") downloadCsv();
    });
  }
  // Linked hover: the row under the pointer names its method to the host, and a method hovered
  // in another chart of the page marks its rows here.
  tblwrap.addEventListener("mouseover", ev => {
    const row = ev.target.closest("tbody tr");
    postHover(row ? row.dataset.m : null);
  });
  tblwrap.addEventListener("mouseleave", () => postHover(null));
  // A click on a row follows its method on every chart of the page; its link still opens.
  tblwrap.addEventListener("click", ev => {
    const row = ev.target.closest("tbody tr");
    if (row && EMBEDDED && !ev.target.closest("a")) pickMethod(row.dataset.m);
  });
  onHoverMessage(method => {
    hovered = method;
    for (const row of tblwrap.querySelectorAll("tbody tr")) row.classList.toggle("hovered", row.dataset.m === method);
  });
  onFollowMessage(names => {
    followed = names;
    render();
  });
  // Opens with the controls: the filters are the point of this page, not chrome
  // around a figure.
  if (EMBEDDED) document.body.classList.add("embedded");
  setUpPaperView(render, { openInPaper: false });
  buildVariantBtns();
  buildColumnBtns();
  buildChips();
  syncChips();
  syncVariantBtns();
  render();
  window.addEventListener("resize", postHeight);
})();
</script>
</body>
</html>
"""
