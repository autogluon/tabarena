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
toggle buttons, a metric selector (e.g. Improvability vs Elo) that re-anchors
the Pareto front, hover tooltips with exact values, dashed-ring marking of
partially imputed methods, and a collapsible data table.
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
</style>
</head>
<body>
  <div class="viewbar">
    <button class="btn" id="btn-paper" title="White background, chart and legend only — for slides and papers">Paper view</button>
  </div>
  <p class="explorer-title" id="title"></p>
  <div class="explorer-grid" id="grid">
    <div class="sidebox">
      <div class="controls">
        <label class="metricpick" id="metricpick" hidden>Y-axis
          <select id="metric-select"></select>
        </label>
        <div class="btnrow">
          <button class="btn" id="btn-front">Pareto front</button>
          <button class="btn" id="btn-all">All</button>
          <button class="btn" id="btn-none">Clear</button>
        </div>
        <span class="hint">Click methods or family buttons to highlight &middot; hover points for details</span>
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

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;

  // Column order: chips/controls left of the chart, or mirrored.
  document.getElementById("grid").classList.add(CONFIG.chipsSide === "right" ? "chips-right" : "chips-left");

  const svg = document.getElementById("chart");
  const box = document.getElementById("chartbox");
  const chipsBox = document.getElementById("chips");
  const controlsBox = document.querySelector(".controls");
  const tip = makeTooltip(box);
  // Matches the point labels drawn in `render` — see the side/overlap logic there.
  const textWidth = makeTextMeasurer(svg, { size: 13, weight: 700 });

  // Marker glyph per variant at (cx, cy); trajectories use circles everywhere.
  function drawMark(parent, cx, cy, variant, color, size, opacity, dataM, whiteStroke) {
    const common = { opacity: opacity, "data-m": dataM };
    let node;
    if (TRAJECTORY || variant === "Default" || !variant) {
      node = el("circle", { ...common, cx, cy, r: size, fill: color }, parent);
    } else if (variant === "Tuned") {
      const s = size * 1.75;
      node = el("rect", { ...common, x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1.5, fill: color }, parent);
    } else if (variant === "Tuned + Ens.") {
      const d = size * 0.95;
      node = el("path", {
        ...common,
        d: `M${cx - d},${cy - d} L${cx + d},${cy + d} M${cx - d},${cy + d} L${cx + d},${cy - d}`,
        stroke: color, "stroke-width": size * 0.62, fill: "none", "stroke-linecap": "round",
      }, parent);
    } else {
      // Any other variant (e.g. "Baseline", holdout types): diamond.
      const s = size * 1.45;
      node = el("rect", {
        ...common, x: cx - s / 2, y: cy - s / 2, width: s, height: s, rx: 1,
        fill: color, transform: `rotate(45 ${cx} ${cy})`,
      }, parent);
    }
    if (whiteStroke && variant !== "Tuned + Ens.") {
      node.setAttribute("stroke", "var(--card)");
      node.setAttribute("stroke-width", "1");
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
  const byMethod = new Map();
  for (const p of POINTS) {
    if (!byMethod.has(p.method)) byMethod.set(p.method, []);
    byMethod.get(p.method).push(p); // insertion order = builder's point order
  }

  const METRICS = CONFIG.metrics;
  let metricKey = METRICS[0].key;
  const metricByKey = {};
  for (const m of METRICS) metricByKey[m.key] = m;

  // Single time axis per explorer (the scatter ships inference time, the
  // trajectories train time).
  const X_AXIS = CONFIG.xAxes[0];
  const xKey = X_AXIS.key;

  function mval(p, metric) { return p[metric.key]; }

  function computeFront(metric) {
    const xk = xKey;
    const pts = [...POINTS].sort((a, b) =>
      a[xk] - b[xk] || (metric.lowerBetter ? mval(a, metric) - mval(b, metric) : mval(b, metric) - mval(a, metric)));
    const verts = [];
    const methods = new Set();
    let best = null;
    for (const p of pts) {
      const v = mval(p, metric);
      if (best === null || (metric.lowerBetter ? v < best : v > best)) {
        if (best !== null) verts.push([p[xk], best]);
        verts.push([p[xk], v]);
        best = v;
        methods.add(p.method);
      }
    }
    return { verts, methods };
  }

  // The chart opens on the front, and keeps following it while the metric changes:
  // each metric has its own front (a method can lead on relative gain and be
  // mid-field on Elo), so a set carried over from the previous metric would leave
  // methods drawn on the front but greyed out. `followFront` drops as soon as the
  // reader picks methods themselves, so their selection is never overwritten.
  const state = { active: new Set(computeFront(metricByKey[metricKey]).methods), followFront: true };

  // ---------- chart ----------
  // A flat, fixed-height plot: two of these panels then fit on one screen.
  const CHART_H = 400;
  const M = { l: 62, r: 18, t: 14, b: 52 };

  function render() {
    const metric = metricByKey[metricKey];
    svg.textContent = "";
    const W = Math.max(360, Math.round(box.clientWidth));
    const H = CHART_H;
    svg.setAttribute("width", W);
    svg.setAttribute("height", H);
    // Keep the chip list from outgrowing the chart beside it.
    chipsBox.style.maxHeight = Math.max(170, H - controlsBox.offsetHeight + 20) + "px";

    // x scale (log)
    const xsAll = POINTS.map(p => p[xKey]);
    const xmin = Math.min(...xsAll) * 0.65, xmax = Math.max(...xsAll) * 1.6;
    const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
    const X = v => M.l + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - M.l - M.r);

    const vals = POINTS.map(p => mval(p, metric));
    let y0, y1;
    if (metric.fromZero) {
      y0 = 0; y1 = Math.max(...vals) * 1.07;
    } else {
      const pad = (Math.max(...vals) - Math.min(...vals)) * 0.07;
      y0 = Math.min(...vals) - pad; y1 = Math.max(...vals) + pad;
    }
    const Y = v => M.t + (1 - (v - y0) / (y1 - y0)) * (H - M.t - M.b);

    // grid + axes
    const grid = el("g", {}, svg);
    for (let e = Math.ceil(lx0); Math.pow(10, e) < xmax; e++) {
      const gx = X(Math.pow(10, e));
      el("line", { x1: gx, y1: M.t, x2: gx, y2: H - M.b, stroke: "var(--line)", "stroke-width": 1 }, grid);
      const lbl = fmtNum(Math.pow(10, e), e >= 0 ? 0 : -e);
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
    el("text", {
      x: (M.l + W - M.r) / 2, y: H - 10, "text-anchor": "middle", "font-size": 14,
      "font-weight": 650, fill: "var(--ink)",
    }, grid).textContent = X_AXIS.axisLabel;
    el("text", {
      x: 0, y: 0, "text-anchor": "middle", "font-size": 14, "font-weight": 650, fill: "var(--ink)",
      transform: `translate(16 ${(M.t + H - M.b) / 2}) rotate(-90)`,
    }, grid).textContent = metric.axisLabel;

    drawOptimalArrow(grid, metric.lowerBetter, M, W, H);

    // pareto front (always shown)
    const front = computeFront(metric);
    const fv = front.verts;
    if (fv.length) {
      let d = `M${X(fv[0][0])},${metric.lowerBetter ? M.t : H - M.b}`;
      for (const [fx, fy] of fv) d += ` L${X(fx)},${Y(fy)}`;
      d += ` L${W - M.r},${Y(fv[fv.length - 1][1])}`;
      el("path", { d, fill: "none", stroke: "var(--ink)", "stroke-width": 1.6, "stroke-dasharray": "7 5", opacity: 0.85 }, svg);
    }

    const isOn = m => state.active.has(m);

    // connectors: variant links (scatter) / the trajectory itself
    const conn = el("g", {}, svg);
    for (const [method, pts] of byMethod) {
      if (pts.length < 2) continue;
      const on = isOn(method);
      if (!TRAJECTORY && !on) continue; // scatter: connectors only for active methods
      const dd = pts.map((p, i) => `${i ? "L" : "M"}${X(p[xKey])},${Y(mval(p, metric))}`).join(" ");
      el("path", {
        d: dd, fill: "none",
        stroke: on ? FAM_VAR[pts[0].family] : "var(--pt-muted)",
        "stroke-width": on ? (TRAJECTORY ? 2 : 1.4) : 1,
        opacity: on ? 0.6 : 0.35,
        "data-m": method,
      }, conn);
    }

    // points: inactive first, active on top
    const ptsOff = el("g", {}, svg);
    const ptsOn = el("g", {}, svg);
    for (const [method, pts] of byMethod) {
      const on = isOn(method);
      for (const p of pts) {
        const color = on ? FAM_VAR[p.family] : "var(--pt-muted)";
        const size = (on ? 7 : 5) * (TRAJECTORY ? 0.8 : 1);
        const op = on ? 0.95 : 0.5;
        drawMark(on ? ptsOn : ptsOff, X(p[xKey]), Y(mval(p, metric)), p.variant, color, size, op, p.method, on);
        // Imputation ring: every affected point in scatter mode; only the
        // trajectory's end point in trajectory mode (a ring on all ~8 line
        // points would read as beads, and the chip's ‡ already flags the line).
        if (p.imputed && (!TRAJECTORY || p === pts[pts.length - 1])) {
          drawImputedRing(on ? ptsOn : ptsOff, X(p[xKey]), Y(mval(p, metric)), size, color, op, p.method);
        }
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
      const px = X(best[xKey]);
      const w = textWidth(method);
      const toRight = px + 10 + w <= W - M.r;
      labels.push({
        method, family: best.family, w,
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
        fill: FAM_VAR[l.family], "paint-order": "stroke", stroke: "var(--card)", "stroke-width": 3.5,
        "text-anchor": l.anchor,
      }, lg);
      t.textContent = l.method;
    }

    // invisible hit targets on top (bigger than marks)
    const hits = el("g", {}, svg);
    for (const p of POINTS) {
      const h = el("circle", { cx: X(p[xKey]), cy: Y(mval(p, metric)), r: 12, fill: "transparent", cursor: "pointer" }, hits);
      h.addEventListener("mouseenter", ev => showTip(p, ev));
      h.addEventListener("mousemove", ev => tip.move(ev));
      h.addEventListener("mouseleave", () => hideTip(p.method));
      h.addEventListener("click", () => toggle(p.method));
    }
  }

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
    let html = `<div class="t-name">${p.method}` + (sub ? ` <span class="t-var">(${sub})</span>` : "") + "</div>" +
      `<div>${p.family}</div>`;
    for (const m of METRICS) {
      html += `<div>${m.label}: <b>${fmtMetric(m, mval(p, m))}</b></div>`;
    }
    html += `<div>${X_AXIS.short}: <b>${fmtTime(p[xKey])}</b></div>`;
    if (p.imputed) html += `<div class="t-imp">Imputed on ${fmtNum(p.imputed_pct, 0)}% of datasets</div>`;
    tip.show(html, ev);
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
        const b = document.createElement("button");
        b.className = "chip";
        b.style.setProperty("--fam", FAM_VAR[fam]);
        const imputed = byMethod.get(m).some(p => p.imputed);
        const label = document.createElement("span");
        label.textContent = m;
        b.appendChild(Object.assign(document.createElement("span"), { className: "dot" }));
        b.appendChild(label);
        if (imputed) {
          const mark = document.createElement("span");
          mark.className = "imp-mark";
          mark.textContent = "‡";
          b.appendChild(mark);
        }
        b.title = m + (imputed ? " — partially imputed" : "");
        b.addEventListener("click", () => toggle(m));
        set.appendChild(b);
        chipByMethod.set(m, b);
      }
      row.appendChild(set);
      chipsBox.appendChild(row);
    }
  }
  function syncChips() {
    for (const [m, b] of chipByMethod) b.setAttribute("aria-pressed", String(state.active.has(m)));
    for (const [fam, b] of famChips) {
      b.setAttribute("aria-pressed", String(familyMethods(fam).every(m => state.active.has(m))));
    }
  }
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
  // Switching the y-axis moves the front, so the highlighted set moves with it.
  function setMetric(key) {
    metricKey = key;
    if (state.followFront) showFront(); else render();
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

  // Embedded, the host can pick the metric for us: the leaderboard's "I care about" control
  // decides whether the page leads with Elo or Improvability, and every panel follows without
  // regenerating an artifact per metric. Ignored when the metric is not one this chart offers.
  window.addEventListener("message", ev => {
    const d = ev.data;
    if (!d || d.type !== "tabarena-explorer-metric") return;
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
    } else {
      html += '<span class="item"><svg width="26" height="8" viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="var(--muted)" stroke-width="2"/><circle cx="6" cy="4" r="2.6" fill="var(--muted)"/><circle cx="16" cy="4" r="2.6" fill="var(--muted)"/></svg> Tuning trajectory (more configs &rarr; more time)</span>';
    }
    html += '<span class="item"><svg width="26" height="8" viewBox="0 0 26 8"><line x1="0" y1="4" x2="26" y2="4" stroke="var(--ink)" stroke-width="1.6" stroke-dasharray="6 4"/></svg> Pareto front (always shown)</span>';
    if (POINTS.some(p => p.imputed)) {
      html += '<span class="item"><svg width="18" height="18" viewBox="0 0 18 18"><circle cx="9" cy="9" r="4" fill="var(--muted)"/><circle cx="9" cy="9" r="7.5" fill="none" stroke="var(--muted)" stroke-width="1.3" stroke-dasharray="3 2.5"/></svg> &Dagger; partially imputed</span>';
    }
    // Model family, named: highlighted points are colored by family, and in paper
    // view the chip list that would otherwise decode them is hidden.
    const families = FAM_ORDER.filter(f => POINTS.some(p => p.family === f));
    if (families.length > 1) {
      html += '<span class="legendbreak"></span><span class="item">Family:</span>';
      for (const fam of families) {
        html += `<span class="item"><svg width="12" height="12" viewBox="0 0 12 12">` +
          `<circle cx="6" cy="6" r="5" fill="${FAM_VAR[fam]}"/></svg> ` +
          `<span style="color:${FAM_INK[fam]}">${fam}</span></span>`;
      }
    }
    box2.innerHTML = html;
  }

  // ---------- paper view ----------
  setUpPaperView(render);
  setUpExport(() => [{ svg: svg, dx: 0 }], () => slugify(document.title));

  // ---------- data table ----------
  function buildTable() {
    const m0 = metricByKey[METRICS[0].key];
    const rows = [...POINTS].sort((a, b) =>
      m0.lowerBetter ? mval(a, m0) - mval(b, m0) : mval(b, m0) - mval(a, m0));
    let html = "<table><thead><tr><th>Method</th>";
    html += TRAJECTORY ? "<th>Configs</th>" : "<th>Variant</th>";
    html += "<th>Family</th>";
    for (const m of METRICS) html += `<th>${m.label}</th>`;
    html += `<th>${X_AXIS.short}</th><th>Imputed</th></tr></thead><tbody>`;
    for (const p of rows) {
      html += `<tr><td>${p.method}</td><td>${TRAJECTORY ? (p.n_configs != null ? p.n_configs : "—") : p.variant}</td><td>${p.family}</td>`;
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
  let resizeTimer = null;
  window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(render, 120);
  });
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
