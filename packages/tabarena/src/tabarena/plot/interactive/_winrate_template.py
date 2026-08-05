"""HTML/CSS/JS template for the interactive win-rate matrix.

The interactive twin of the static ``winrate_matrix`` figure: the same square
heatmap of pairwise win rates, plus hover values, two sets of toggles (models by
family, and tuning variants), a reordering control, paper view and SVG/PNG/PDF
export.

The heatmap is live SVG, which is what lets the shared export helpers build a
figure file from it (see
:func:`tabarena.plot.interactive._explorer_shared.EXPLORER_BASE_JS`). The colour
scale is drawn into that SVG rather than as HTML beside it, so it survives paper
view and lands in the exported figure.

Kept as a Python string constant (rather than a package-data file) so it ships
with the package without any build-system data-file configuration.
:func:`tabarena.plot.interactive.winrate_explorer.build_winrate_explorer_html`
substitutes the placeholders (see
:func:`tabarena.plot.interactive._explorer_shared.render_explorer_html`).
"""

from __future__ import annotations

WINRATE_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  /* The matrix scrolls in both directions: 40+ models will not fit a page either
     way, and shrinking the cells to fit makes the numbers unreadable. `margin:
     auto` centres it once the selection is small enough to fit — and collapses
     to zero when it overflows, which `justify-content: center` would not. */
  .wr-scroll {
    overflow: auto; max-width: 100%;
    scrollbar-width: thin; scrollbar-color: var(--pt-muted) transparent;
  }
  .wr-scroll::-webkit-scrollbar { width: 11px; height: 11px; }
  .wr-scroll::-webkit-scrollbar-thumb {
    background: var(--pt-muted); border-radius: 8px;
    border: 3px solid transparent; background-clip: content-box;
  }
  #chart { display: block; margin-inline: auto; }
  .wr-empty { padding: 40px 0; text-align: center; color: var(--muted); font-size: 13px; }
  .chips-head { flex: 1 1 100%; font-size: 12.5px; color: var(--muted); font-weight: 600; }
  .chips { margin: 2px 0 10px; flex-direction: row; flex-wrap: wrap; gap: 12px 28px; align-items: flex-start; }
  .chiprow { flex: 0 1 auto; min-width: 0; }
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, matrix only — for slides and papers">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>
  <div class="controls">
    <button class="btn toggle" id="btn-best"
      title="Keep only each model's best-performing variant, one row and column per model — as the static figure does">One per model</button>
    <span class="grouplabel">Variants</span>
    <div class="btnrow" id="variant-btns"></div>
    <label class="metricpick">Order
      <select id="order-select">
        <option value="published">As published</option>
        <option value="winrate">Mean win rate</option>
        <option value="name">Name</option>
      </select>
    </label>
    <label class="metricpick">Size
      <select id="zoom-select" title="Fit scales the whole matrix into the panel; the percentages draw it at reading size and scroll">
        <option value="fit">Fit panel</option>
        <option value="0.6">60%</option>
        <option value="0.8">80%</option>
        <option value="1">100%</option>
      </select>
    </label>
    <span class="hint">Click a row or column label to remove that model</span>
  </div>
  <div class="exportbar" id="exportbar" hidden>
    <span class="hint">Export figure</span>
    <button class="btn" id="btn-svg" title="Download as SVG — vector, keeps text selectable">Download SVG</button>
    <button class="btn" id="btn-pdf" title="Download as a one-page PDF">Download PDF</button>
    <button class="btn" id="btn-png" title="Download as PNG at 3x scale">Download PNG</button>
  </div>
  <div class="chips" id="chips"></div>
  <div class="wr-scroll" id="wrap">
    <svg id="chart" role="img" aria-label="Pairwise win-rate matrix"></svg>
  </div>
  <div class="tooltip"></div>

<script>
(function () {
  "use strict";
  const CONFIG = __CONFIG_JSON__;
  const POINTS = __POINTS_JSON__;

__BASE_JS__

  // POINTS is one record per matrix label (its model, variant, family and mean
  // win rate); the matrix itself is CONFIG.matrix, row-major over CONFIG.methods.
  const METHODS = CONFIG.methods;
  const MATRIX = CONFIG.matrix;
  const rowOf = new Map(METHODS.map((m, i) => [m, i]));
  const info = new Map(POINTS.map(p => [p.method, p]));
  const MODELS = [...new Set(POINTS.map(p => p.model))];
  const VARIANTS = CONFIG.variants.filter(v => POINTS.some(p => p.variant === v));
  const VARIANT_BTN = { "Tuned + Ens.": "Tuned + Ensembled" };

  // Nothing is hidden on load. Which entrants compete is now decided upstream by the
  // leaderboard's entrant pool (see tabarena.evaluation.entrants): if a system is in this
  // matrix at all, the reader picked a pool that includes it, and its win-rates were
  // computed against that field. Hiding it here would contradict that choice.
  const OFF_BY_DEFAULT = new Set([]);
  const familyOf = new Map(POINTS.map(p => [p.model, p.family]));
  const DEFAULT_MODELS = MODELS.filter(m => !OFF_BY_DEFAULT.has(familyOf.get(m)));

  const state = {
    models: new Set(DEFAULT_MODELS.length > 1 ? DEFAULT_MODELS : MODELS),
    variants: new Set(VARIANTS),
    // One entry per model by default, as the static figure shows it: 80 rows of
    // model-variant pairs is a wall, and the comparison people want first is
    // between models.
    best: true,
    order: "published",
    // Default to fitting the panel: at reading size the matrix is taller than a
    // screen, and the shape of the heatmap is what most readers want first.
    zoom: "fit",
  };
  // Ceiling on the fitted height. Fitting is normally width-driven; this only bites
  // on very tall selections, and it is set where the labels are still readable
  // rather than at half a screen, which would shrink 30 rows to 8px type.
  const FIT_H = 1040;
  // Nominal (drawn) size of the last render; the display size is a CSS scale of it.
  let nomW = 0, nomH = 0;
  const svg = document.getElementById("chart");
  const wrap = document.getElementById("wrap");
  const tooltip = document.querySelector(".tooltip");
  const chipsBox = document.getElementById("chips");
  const famChips = new Map();
  const chipByModel = new Map();

  // Sized against the static figure rather than the other explorers: that one uses
  // 16pt tick labels and 18pt axis labels on generously sized cells, and it stayed
  // the easier of the two to read. Weight still matches the explorers' 650, so a
  // model name looks like the same name everywhere.
  const LABEL_SIZE = 16, LABEL_WEIGHT = 650, CAPTION_SIZE = 18, CELL = 42, VALUE_SIZE = 15;
  // The colour key is a vertical bar to the right of the matrix, as on the static
  // figure. It sits beside the rows rather than above them, so it is in view
  // wherever the reader is in a tall matrix.
  const BAR_W = 30, BAR_GAP = 32, TICK_SIZE = 15;
  const SCALE_CAPTION = "win rate of the row over the column";
  const SCALE_CAPTION_SHORT = "win rate";

  // Purple (the column wins) through white to green (the row wins), the same
  // diverging reading as the static figure's PRGn colormap.
  const LOSE = [118, 42, 131], MID = [247, 247, 247], WIN = [27, 120, 55];
  function cellColor(rate) {
    if (rate == null || !isFinite(rate)) return "#8884";
    const t = Math.max(0, Math.min(1, rate));
    const [a, b, u] = t < 0.5 ? [LOSE, MID, t / 0.5] : [MID, WIN, (t - 0.5) / 0.5];
    return "rgb(" + a.map((v, i) => Math.round(v + u * (b[i] - v))).join(",") + ")";
  }

  // Measured where possible: the labels are model names of wildly different
  // lengths, and the margins have to clear the longest one exactly or a rotated
  // column label runs into the caption above it. `getComputedTextLength` returns 0
  // when the frame has not been laid out yet (it is lazily loaded, and may still be
  // off-screen on the first paint), which would collapse both margins and push
  // every label outside the viewBox — invisible until something forced a redraw.
  // So fall back to an estimate, and redraw once the fonts have settled.
  function textWidth(text, size, weight) {
    const probe = el("text", {
      "font-size": size || LABEL_SIZE, "font-weight": weight || LABEL_WEIGHT,
    }, svg);
    probe.textContent = text;
    let width = 0;
    try { width = probe.getComputedTextLength(); } catch (e) { width = 0; }
    probe.remove();
    return width > 0 ? width : text.length * (size || LABEL_SIZE) * 0.58;
  }

  // Labels always carry their variant tag — "(default)" / "(tuned + ensembled)" —
  // including in one-per-model mode, where the tag is precisely what says which
  // variant survived the filter. The static figure labels them the same way.
  function labelText(label) {
    return label;
  }

  function shown() {
    let list = METHODS.filter(label => {
      const p = info.get(label);
      if (!p || !state.models.has(p.model)) return false;
      return !p.variant || state.variants.has(p.variant);
    });
    if (state.best) {
      // Best *among the variants still selected*, so the variant toggles keep
      // meaning something in this mode.
      const pick = new Map();
      for (const label of list) {
        const p = info.get(label);
        const held = pick.get(p.model);
        if (!held || (info.get(held).mean || 0) < (p.mean || 0)) pick.set(p.model, label);
      }
      list = list.filter(label => pick.get(info.get(label).model) === label);
    }
    if (state.order === "winrate") {
      list.sort((a, b) => (info.get(b).mean || 0) - (info.get(a).mean || 0));
    } else if (state.order === "name") {
      list.sort((a, b) => labelText(a).localeCompare(labelText(b)));
    }
    return list;
  }

  function render() {
    const list = shown();
    svg.innerHTML = "";
    if (list.length < 2) {
      svg.setAttribute("width", 0);
      svg.setAttribute("height", 0);
      // Also drop the display scale, or the empty chart keeps the last figure's box.
      svg.style.width = svg.style.height = "";
      nomW = nomH = 0;
      wrap.querySelector(".wr-empty") ||
        wrap.insertAdjacentHTML("beforeend",
          '<p class="wr-empty">Select at least two models to compare.</p>');
      postHeight();
      return;
    }
    const empty = wrap.querySelector(".wr-empty");
    if (empty) empty.remove();

    const widths = list.map(label => textWidth(labelText(label)));
    const longest = Math.max(...widths);
    // A label rotated -60° rises 0.866 of its length; the captions sit in the
    // corner box rather than above the columns, where long names would cross them.
    // Floor the header so the two corner captions always have room, even when
    // every selected name is short.
    const headerH = Math.max(Math.ceil(longest * 0.866) + 14, 2 * CAPTION_SIZE + 26);
    const labelW = Math.ceil(longest) + 16;
    const n = list.length;
    // A -60° label anchored at (x, y) reaches (x + 0.5L, y - 0.866L): it grows up
    // *and right*. headerH covers the rise; this covers the run, or the rightmost
    // column's name is clipped by half its length.
    const overhang = Math.ceil(longest * 0.5);
    const matrixRight = labelW + n * CELL;
    // The key occupies the band to the right of the matrix; the label overhang is
    // above the matrix, so the two never collide and the wider of the two wins.
    const tickW = Math.ceil(textWidth("100%", TICK_SIZE, 700));
    const barX = matrixRight + BAR_GAP;
    const keyRight = barX + BAR_W + 12 + tickW + 10 + CAPTION_SIZE;
    const width = Math.max(matrixRight + overhang + 12, keyRight + 10);
    const height = headerH + n * CELL + 12;
    svg.setAttribute("width", width);
    svg.setAttribute("height", height);
    svg.setAttribute("viewBox", "0 0 " + width + " " + height);

    // Captions, stacked in the corner box and bold, as on the static figure.
    const capB = el("text", {
      x: 4, y: headerH - 34, "font-size": CAPTION_SIZE, "font-weight": 700, fill: "var(--ink)",
    }, svg);
    capB.textContent = "Model B: loser →";
    const capA = el("text", {
      x: 4, y: headerH - 14, "font-size": CAPTION_SIZE, "font-weight": 700, fill: "var(--ink)",
    }, svg);
    capA.textContent = "Model A: winner ↓";

    list.forEach((label, j) => {
      const x = labelW + j * CELL + CELL / 2 + 4;
      const text = el("text", {
        x: x, y: headerH - 8, "font-size": LABEL_SIZE, "font-weight": LABEL_WEIGHT,
        "text-anchor": "start", fill: FAM_INK[info.get(label).family] || "var(--ink)",
        transform: "rotate(-60 " + x + " " + (headerH - 8) + ")", cursor: "pointer",
      }, svg);
      text.textContent = labelText(label);
      text.addEventListener("click", () => toggleModel(info.get(label).model));
    });

    list.forEach((rowLabel, i) => {
      const y = headerH + i * CELL;
      const text = el("text", {
        x: labelW - 12, y: y + CELL / 2 + 6, "font-size": LABEL_SIZE, "font-weight": LABEL_WEIGHT,
        "text-anchor": "end", fill: FAM_INK[info.get(rowLabel).family] || "var(--ink)",
        cursor: "pointer",
      }, svg);
      text.textContent = labelText(rowLabel);
      text.addEventListener("click", () => toggleModel(info.get(rowLabel).model));

      list.forEach((colLabel, j) => {
        const rate = valueAt(rowLabel, colLabel);
        const x = labelW + j * CELL;
        const same = rowLabel === colLabel;
        const rect = el("rect", {
          x: x, y: y, width: CELL - 1, height: CELL - 1, rx: 2,
          fill: same ? "#8883" : cellColor(rate),
        }, svg);
        if (same) return;
        const value = el("text", {
          x: x + (CELL - 1) / 2, y: y + CELL / 2 + 5.5, "font-size": VALUE_SIZE,
          "font-weight": 600, "text-anchor": "middle",
          fill: Math.abs(rate - 0.5) > 0.3 ? "#ffffff" : "#14161a", "pointer-events": "none",
        }, svg);
        value.textContent = rate == null || !isFinite(rate) ? "" : Math.round(rate * 100);
        rect.addEventListener("mousemove", ev => showTip(ev, rowLabel, colLabel, rate));
        rect.addEventListener("mouseleave", () => { tooltip.style.display = "none"; });
      });
    });

    drawScale(barX, headerH, n * CELL - 2, tickW);
    nomW = width;
    nomH = height;
    applyZoom();
  }

  // Scale the figure for display only: the width/height *attributes* stay nominal,
  // so the export helpers keep building full-size figures, and the CSS box drives
  // what the page shows (the viewBox does the scaling, so it stays vector-sharp).
  function applyZoom() {
    if (!nomW || !nomH) return;
    let scale = Number(state.zoom);
    if (!(scale > 0)) {
      const availW = Math.max(240, (wrap.clientWidth || nomW) - 4);
      scale = Math.min(1, availW / nomW, FIT_H / nomH);
    }
    svg.style.width = Math.round(nomW * scale) + "px";
    svg.style.height = Math.round(nomH * scale) + "px";
    postHeight();
  }

  // The colour key, drawn into the figure so it is there in paper view and in an
  // exported SVG/PNG/PDF, not only beside the live chart.
  function drawScale(x, y, barH, tickW) {
    const id = "wr-ramp";
    const grad = el("linearGradient", { id: id, x1: "0", x2: "0", y1: "0", y2: "1" },
      el("defs", {}, svg));
    // Top of the bar is the row winning outright, the bottom is the column winning.
    for (const [offset, rate] of [[0, 1], [0.5, 0.5], [1, 0]]) {
      el("stop", { offset: offset, "stop-color": cellColor(rate) }, grad);
    }
    el("rect", {
      x: x, y: y, width: BAR_W, height: barH, rx: 7,
      fill: "url(#" + id + ")", stroke: "var(--muted)", "stroke-width": 1,
    }, svg);
    for (const [frac, label] of [[0, "100%"], [0.5, "50%"], [1, "0%"]]) {
      const ty = y + frac * barH;
      el("line", {
        x1: x + BAR_W, y1: ty, x2: x + BAR_W + 6, y2: ty,
        stroke: "var(--muted)", "stroke-width": 1.5,
      }, svg);
      const tick = el("text", {
        x: x + BAR_W + 11, y: ty + TICK_SIZE * 0.36, "font-size": TICK_SIZE,
        "font-weight": 700, "text-anchor": "start", fill: "var(--ink)",
      }, svg);
      tick.textContent = label;
    }
    // The label reads down the bar. It shortens rather than growing past the bar
    // when only a couple of models are selected: anything above the bar would run
    // into the rightmost column's rotated name.
    const size = CAPTION_SIZE - 2;
    const cx = x + BAR_W + 11 + tickW + 10 + size * 0.5;
    const cy = y + barH / 2;
    const caption = el("text", {
      x: cx, y: cy, "font-size": size, "font-weight": 700,
      "text-anchor": "middle", fill: "var(--ink)",
      transform: "rotate(90 " + cx + " " + cy + ")",
    }, svg);
    caption.textContent =
      textWidth(SCALE_CAPTION, size, 700) <= barH ? SCALE_CAPTION : SCALE_CAPTION_SHORT;
  }

  function valueAt(rowLabel, colLabel) {
    const i = rowOf.get(rowLabel), j = rowOf.get(colLabel);
    if (i == null || j == null) return null;
    const row = MATRIX[i];
    return row ? row[j] : null;
  }

  function showTip(ev, rowLabel, colLabel, rate) {
    tooltip.style.display = "block";
    tooltip.innerHTML =
      '<div class="t-name">' + rowLabel + " vs " + colLabel + "</div>" +
      "<div>" + rowLabel + " wins " + fmtNum(rate * 100, 1) + "% of tasks</div>" +
      '<div class="t-var">' + colLabel + " wins " + fmtNum((1 - rate) * 100, 1) + "%</div>";
    const box = document.body.getBoundingClientRect();
    tooltip.style.left = (ev.clientX - box.left + 14) + "px";
    tooltip.style.top = (ev.clientY - box.top + 12) + "px";
  }

  // ---------- model chips, by family ----------
  function familyModels(fam) {
    return [...new Set(POINTS.filter(p => p.family === fam).map(p => p.model))];
  }
  function meanOfModel(model) {
    const values = POINTS.filter(p => p.model === model && isFinite(p.mean)).map(p => p.mean);
    return values.length ? Math.max(...values) : 0;
  }
  function buildChips() {
    const head = document.createElement("div");
    head.className = "chips-head";
    head.textContent = "Models compared — click to remove, click a family to toggle the whole group";
    chipsBox.appendChild(head);
    for (const fam of FAM_ORDER) {
      const members = familyModels(fam);
      if (!members.length) continue;
      members.sort((a, b) => meanOfModel(b) - meanOfModel(a));
      const row = document.createElement("div");
      row.className = "chiprow";
      const famBtn = document.createElement("button");
      famBtn.className = "famchip";
      famBtn.style.setProperty("--fam", FAM_VAR[fam]);
      famBtn.innerHTML = famChipLabel(fam, members.length);
      famBtn.title = "Toggle all " + members.length + " " + fam + " models";
      famBtn.addEventListener("click", () => toggleFamily(fam));
      row.appendChild(famBtn);
      famChips.set(fam, famBtn);
      const set = document.createElement("div");
      set.className = "chipset";
      for (const model of members) {
        const b = document.createElement("button");
        b.className = "chip";
        b.style.setProperty("--fam", FAM_VAR[fam]);
        b.innerHTML = '<span class="dot"></span><span></span>';
        b.lastChild.textContent = model;
        b.title = model + " — best mean win rate " + fmtNum(meanOfModel(model) * 100, 1) + "%";
        b.addEventListener("click", () => toggleModel(model));
        set.appendChild(b);
        chipByModel.set(model, b);
      }
      row.appendChild(set);
      chipsBox.appendChild(row);
    }
  }
  function syncChips() {
    for (const [model, b] of chipByModel) b.setAttribute("aria-pressed", String(state.models.has(model)));
    for (const [fam, b] of famChips) {
      b.setAttribute("aria-pressed", String(familyModels(fam).every(m => state.models.has(m))));
    }
  }
  function toggleModel(model) {
    if (state.models.has(model)) state.models.delete(model); else state.models.add(model);
    syncChips();
    render();
  }
  function toggleFamily(fam) {
    const members = familyModels(fam);
    const allOn = members.every(m => state.models.has(m));
    for (const m of members) { if (allOn) state.models.delete(m); else state.models.add(m); }
    syncChips();
    render();
  }

  // ---------- variant toggles ----------
  const variantBtns = new Map();
  function buildVariantBtns() {
    const box = document.getElementById("variant-btns");
    for (const v of VARIANTS) {
      const b = document.createElement("button");
      b.className = "btn toggle";
      b.innerHTML = '<span class="swatch"></span>';
      // Spelled out: the matrix labels carry the long form the data uses, so the
      // toggle should not be the only place showing the abbreviated key.
      b.appendChild(document.createTextNode(VARIANT_BTN[v] || v));
      b.style.setProperty("--fam", VARIANT_VAR[v] || "var(--accent)");
      b.title = "Show or hide the " + (VARIANT_BTN[v] || v).toLowerCase() + " results";
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
  }

  // ---------- boot ----------
  if (CONFIG.title) document.getElementById("title").textContent = CONFIG.title;
  document.getElementById("zoom-select").addEventListener("change", ev => {
    state.zoom = ev.target.value;
    applyZoom();
  });
  window.addEventListener("resize", applyZoom);
  // Embedded in a lazily shown panel, the first render can measure a zero-width
  // panel; re-fit once it has a width. Width changes only, so the height applyZoom
  // itself produces cannot feed back into another fit.
  if (window.ResizeObserver) {
    let lastWidth = 0;
    new ResizeObserver(() => {
      const w = wrap.clientWidth;
      if (Math.abs(w - lastWidth) < 2) return;
      lastWidth = w;
      applyZoom();
    }).observe(wrap);
  }
  document.getElementById("order-select").addEventListener("change", ev => {
    state.order = ev.target.value;
    render();
  });
  const bestBtn = document.getElementById("btn-best");
  bestBtn.style.setProperty("--fam", "var(--accent)");
  bestBtn.addEventListener("click", () => {
    state.best = !state.best;
    bestBtn.setAttribute("aria-pressed", String(state.best));
    render();
  });
  bestBtn.setAttribute("aria-pressed", String(state.best));
  setUpPaperView(render);
  setUpExport(() => [{ svg: svg, dx: 0 }], () => slugify(CONFIG.title || document.title));
  buildVariantBtns();
  buildChips();
  syncChips();
  syncVariantBtns();
  render();
  // The first render may have had to estimate label widths (see textWidth); redraw
  // with real measurements as soon as the fonts are in.
  if (document.fonts && document.fonts.ready) document.fonts.ready.then(render);
})();
</script>
</body>
</html>
"""
