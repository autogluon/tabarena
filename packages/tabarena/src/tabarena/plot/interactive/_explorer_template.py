"""HTML/CSS/JS template for the self-contained interactive Pareto explorer.

Kept as a Python string constant (rather than a package-data file) so it ships
with the package without any build-system data-file configuration. The
placeholders ``__PAGE_TITLE__``, ``__CONFIG_JSON__`` and ``__POINTS_JSON__``
are substituted by :func:`tabarena.plot.interactive.pareto_explorer.build_pareto_explorer_html`.

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
  :root {
    --paper: #ffffff;
    --card: #ffffff;
    --ink: #14161a;
    --muted: #6d6c65;
    --line: #e4e3db;
    --accent: #2a78d6;
    --chip-bg: #f2f1ec;
    --pt-muted: #b9b8b1;
    --fam-foundation: #2a78d6;
    --fam-tree: #eb6834;
    --fam-nn: #1baf7a;
    --fam-other: #4a3aa7;
    --fam-reference: #e87ba4;
    --fam-baseline: #898781;
    --optimal: #228b22;
    --tooltip-bg: #14161a;
    --tooltip-ink: #fbfbf9;
    color-scheme: light;
  }
  @media (prefers-color-scheme: dark) {
    :root {
      --paper: #131316;
      --card: #1b1b1f;
      --ink: #f0efea;
      --muted: #9b9a92;
      --line: #2e2e33;
      --accent: #3987e5;
      --chip-bg: #232327;
      --pt-muted: #55555c;
      --fam-foundation: #3987e5;
      --fam-tree: #d95926;
      --fam-nn: #199e70;
      --fam-other: #9085e9;
      --fam-reference: #d55181;
      --fam-baseline: #898781;
      --optimal: #2ea043;
      --tooltip-bg: #f0efea;
      --tooltip-ink: #14161a;
      color-scheme: dark;
    }
  }

  html, body { margin: 0; background: var(--paper); }
  body {
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI", sans-serif;
    line-height: 1.5;
    padding: 10px 12px 14px;
  }

  .explorer-title { font-size: 15px; font-weight: 650; margin: 0 0 8px; }
  .controls { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 14px; margin-bottom: 10px; }
  .controls .hint { font-size: 12.5px; color: var(--muted); }
  .btnrow { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn {
    font: 600 12.5px/1 system-ui, sans-serif; color: var(--ink);
    background: var(--chip-bg); border: 1px solid var(--line); border-radius: 7px;
    padding: 6px 11px; cursor: pointer;
  }
  .btn:hover { border-color: var(--muted); }
  .btn:focus-visible, .chip:focus-visible, .famchip:focus-visible, select:focus-visible {
    outline: 2px solid var(--accent); outline-offset: 2px;
  }
  .metricpick { display: inline-flex; align-items: center; gap: 6px; font-size: 12.5px; font-weight: 600; color: var(--muted); }
  .metricpick select {
    font: 600 12.5px/1.2 system-ui, sans-serif; color: var(--ink);
    background: var(--chip-bg); border: 1px solid var(--line); border-radius: 7px;
    padding: 5px 7px; cursor: pointer;
  }

  .chips { display: flex; flex-direction: column; gap: 6px; margin-bottom: 10px; }
  .chiprow { display: flex; align-items: flex-start; gap: 9px; }
  .famchip {
    flex: 0 0 152px; display: inline-flex; align-items: center; gap: 6px; justify-content: flex-end;
    font: 650 10.5px/1.3 system-ui, sans-serif; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); background: var(--chip-bg); border: 1px dashed var(--line);
    border-radius: 999px; padding: 5px 10px; cursor: pointer; margin-top: 1px;
  }
  .famchip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--fam); flex: none; }
  .famchip .count { font-weight: 500; letter-spacing: 0; opacity: 0.75; }
  .famchip:hover { border-color: var(--fam); color: var(--ink); }
  .famchip[aria-pressed="true"] {
    border: 1px solid var(--fam);
    background: color-mix(in srgb, var(--fam) 13%, transparent);
    color: var(--ink);
  }
  .chipset { display: flex; flex-wrap: wrap; gap: 4px; }
  .chip {
    display: inline-flex; align-items: center; gap: 5px;
    font: 500 12.5px/1 system-ui, sans-serif; color: var(--ink);
    background: none; border: 1px solid var(--line); border-radius: 999px;
    padding: 5px 10px 5px 8px; cursor: pointer;
  }
  .chip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--pt-muted); flex: none; }
  .chip .imp-mark { color: var(--muted); font-weight: 700; margin-left: -2px; }
  .chip[aria-pressed="true"] { border-color: var(--fam); background: color-mix(in srgb, var(--fam) 13%, transparent); font-weight: 650; }
  .chip[aria-pressed="true"] .dot { background: var(--fam); }
  .chip:hover { border-color: var(--muted); }

  .chartbox { position: relative; }
  .chartbox svg { width: 100%; height: auto; display: block; }
  .legendstrip {
    display: flex; flex-wrap: wrap; gap: 5px 16px; align-items: center;
    font-size: 12.5px; color: var(--muted); padding: 2px 2px 8px;
  }
  .legendstrip .item { display: inline-flex; align-items: center; gap: 6px; }

  .tooltip {
    position: absolute; pointer-events: none; display: none;
    background: var(--tooltip-bg); color: var(--tooltip-ink);
    border-radius: 8px; padding: 8px 11px; font-size: 12px; line-height: 1.45;
    max-width: 260px; z-index: 5; font-variant-numeric: tabular-nums;
    box-shadow: 0 4px 14px rgba(0,0,0,0.25);
  }
  .tooltip .t-name { font-weight: 700; font-size: 12.5px; }
  .tooltip .t-var { opacity: 0.75; }
  .tooltip .t-imp { opacity: 0.85; font-style: italic; }

  details.datatable { margin-top: 8px; font-size: 12.5px; }
  details.datatable summary { cursor: pointer; color: var(--muted); font-weight: 600; }
  details.datatable .tblwrap { overflow-x: auto; margin-top: 8px; }
  details.datatable table { border-collapse: collapse; font-variant-numeric: tabular-nums; min-width: 560px; }
  details.datatable th, details.datatable td {
    text-align: left; padding: 3px 12px 3px 0; border-bottom: 1px solid var(--line);
  }
  details.datatable th { font-size: 11px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); }

  svg text { font-family: system-ui, -apple-system, "Segoe UI", sans-serif; }

  @media (prefers-reduced-motion: no-preference) {
    .chip, .btn, .famchip { transition: border-color 120ms ease, background-color 120ms ease; }
  }
</style>
</head>
<body>
  <p class="explorer-title" id="title"></p>
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
  <div class="legendstrip" id="legendstrip"></div>
  <div class="chartbox" id="chartbox">
    <svg id="chart" viewBox="0 0 960 540" role="img" aria-label="Pareto front explorer"></svg>
    <div class="tooltip"></div>
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

  const FAM_ORDER = ["Foundation Model", "Tree-based", "Neural Network", "Other", "Reference Pipeline", "Baseline"];
  const FAM_VAR = {
    "Foundation Model": "var(--fam-foundation)",
    "Tree-based": "var(--fam-tree)",
    "Neural Network": "var(--fam-nn)",
    "Other": "var(--fam-other)",
    "Reference Pipeline": "var(--fam-reference)",
    "Baseline": "var(--fam-baseline)",
  };
  const NS = "http://www.w3.org/2000/svg";
  const TRAJECTORY = CONFIG.mode === "trajectory";

  const titleEl = document.getElementById("title");
  if (CONFIG.title) titleEl.textContent = CONFIG.title; else titleEl.hidden = true;

  const svg = document.getElementById("chart");
  const box = document.getElementById("chartbox");
  const tipEl = box.querySelector(".tooltip");
  const tip = {
    show(html, ev) { tipEl.innerHTML = html; tipEl.style.display = "block"; this.move(ev); },
    move(ev) {
      const r = box.getBoundingClientRect();
      let tx = ev.clientX - r.left + 14, ty = ev.clientY - r.top + 12;
      if (tx > r.width - 270) tx = ev.clientX - r.left - 274;
      tipEl.style.left = tx + "px";
      tipEl.style.top = ty + "px";
    },
    hide() { tipEl.style.display = "none"; },
  };

  // ---------- helpers ----------
  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

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

  function ticks(min, max, target) {
    const raw = (max - min) / target;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    let step = mag;
    for (const m of [1, 2, 5, 10]) {
      if (mag * m >= raw) { step = mag * m; break; }
    }
    const out = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) out.push(v);
    return out;
  }

  function fmtTime(v) {
    if (v >= 100) return v.toFixed(0) + " s";
    if (v >= 1) return v.toFixed(1) + " s";
    if (v >= 0.1) return v.toFixed(2) + " s";
    return v.toFixed(3) + " s";
  }

  function fmtMetric(metric, v) {
    return v.toFixed(metric.decimals) + (metric.suffix || "");
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

  function mval(p, metric) { return p[metric.key]; }

  function computeFront(metric) {
    const pts = [...POINTS].sort((a, b) =>
      a.x - b.x || (metric.lowerBetter ? mval(a, metric) - mval(b, metric) : mval(b, metric) - mval(a, metric)));
    const verts = [];
    const methods = new Set();
    let best = null;
    for (const p of pts) {
      const v = mval(p, metric);
      if (best === null || (metric.lowerBetter ? v < best : v > best)) {
        if (best !== null) verts.push([p.x, best]);
        verts.push([p.x, v]);
        best = v;
        methods.add(p.method);
      }
    }
    return { verts, methods };
  }

  const state = { active: new Set(computeFront(metricByKey[metricKey]).methods) };

  // ---------- chart ----------
  const W = 960, H = 540, M = { l: 62, r: 18, t: 14, b: 52 };
  const xsAll = POINTS.map(p => p.x);
  const xmin = Math.min(...xsAll) * 0.65, xmax = Math.max(...xsAll) * 1.6;
  const lx0 = Math.log10(xmin), lx1 = Math.log10(xmax);
  const X = v => M.l + (Math.log10(v) - lx0) / (lx1 - lx0) * (W - M.l - M.r);

  function render() {
    const metric = metricByKey[metricKey];
    svg.textContent = "";

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
      const lbl = e >= 0 ? String(Math.pow(10, e)) : Math.pow(10, e).toFixed(-e);
      el("text", { x: gx, y: H - M.b + 20, "text-anchor": "middle", "font-size": 12, fill: "var(--muted)" }, grid)
        .textContent = lbl;
    }
    for (const yv of ticks(y0, y1, 6)) {
      const gy = Y(yv);
      el("line", { x1: M.l, y1: gy, x2: W - M.r, y2: gy, stroke: "var(--line)", "stroke-width": 1 }, grid);
      el("text", { x: M.l - 8, y: gy + 4, "text-anchor": "end", "font-size": 12, fill: "var(--muted)" }, grid)
        .textContent = yv;
    }
    el("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "none", stroke: "var(--line)" }, grid);
    el("text", { x: (M.l + W - M.r) / 2, y: H - 10, "text-anchor": "middle", "font-size": 14, fill: "var(--ink)" }, grid)
      .textContent = CONFIG.x_label;
    el("text", { x: 0, y: 0, "text-anchor": "middle", "font-size": 14, fill: "var(--ink)",
      transform: `translate(16 ${(M.t + H - M.b) / 2}) rotate(-90)` }, grid).textContent = metric.axisLabel;

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
      const dd = pts.map((p, i) => `${i ? "L" : "M"}${X(p.x)},${Y(mval(p, metric))}`).join(" ");
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
        drawMark(on ? ptsOn : ptsOff, X(p.x), Y(mval(p, metric)), p.variant, color, size, op, p.method, on);
        if (p.imputed) drawImputedRing(on ? ptsOn : ptsOff, X(p.x), Y(mval(p, metric)), size, color, op, p.method);
      }
    }

    // labels for active methods at their best point, greedy de-overlap
    const labels = [];
    for (const [method, pts] of byMethod) {
      if (!isOn(method)) continue;
      const best = pts.reduce((a, b) =>
        (metric.lowerBetter ? mval(a, metric) < mval(b, metric) : mval(a, metric) > mval(b, metric)) ? a : b);
      labels.push({ method, family: best.family, x: X(best.x) + 10, y: Y(mval(best, metric)) - 10 });
    }
    labels.sort((a, b) => a.y - b.y);
    for (let i = 1; i < labels.length; i++) {
      for (let j = 0; j < i; j++) {
        if (Math.abs(labels[i].x - labels[j].x) < 110 && Math.abs(labels[i].y - labels[j].y) < 15) {
          labels[i].y = labels[j].y + 15;
        }
      }
    }
    const lg = el("g", {}, svg);
    for (const l of labels) {
      const t = el("text", {
        x: Math.min(l.x, W - M.r - 8), y: Math.max(l.y, M.t + 12), "font-size": 13, "font-weight": 700,
        fill: FAM_VAR[l.family], "paint-order": "stroke", stroke: "var(--card)", "stroke-width": 3.5,
        "text-anchor": l.x > W - 120 ? "end" : "start",
      }, lg);
      t.textContent = l.method;
    }

    // invisible hit targets on top (bigger than marks)
    const hits = el("g", {}, svg);
    for (const p of POINTS) {
      const h = el("circle", { cx: X(p.x), cy: Y(mval(p, metric)), r: 12, fill: "transparent", cursor: "pointer" }, hits);
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
    html += `<div>${CONFIG.x_short}: <b>${fmtTime(p.x)}</b></div>`;
    if (p.imputed) html += `<div class="t-imp">Imputed on ${p.imputed_pct.toFixed(0)}% of datasets</div>`;
    tip.show(html, ev);
  }
  function hideTip(method) {
    emphasize(method, false);
    tip.hide();
  }

  // ---------- chips ----------
  const chipsBox = document.getElementById("chips");
  const chipByMethod = new Map();
  const famChips = new Map();
  function familyMethods(fam) {
    return [...byMethod.keys()].filter(m => byMethod.get(m)[0].family === fam);
  }
  function buildChips() {
    for (const fam of FAM_ORDER) {
      const methods = familyMethods(fam).sort();
      if (!methods.length) continue;
      const row = document.createElement("div");
      row.className = "chiprow";
      const famBtn = document.createElement("button");
      famBtn.className = "famchip";
      famBtn.style.setProperty("--fam", FAM_VAR[fam]);
      famBtn.innerHTML = `<span class="dot"></span>${fam} <span class="count">&times;${methods.length}</span>`;
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
    syncChips();
    render();
  }
  function toggleFamily(fam) {
    const methods = familyMethods(fam);
    const allOn = methods.every(m => state.active.has(m));
    for (const m of methods) {
      if (allOn) state.active.delete(m); else state.active.add(m);
    }
    syncChips();
    render();
  }
  function setActive(methods) {
    state.active = new Set(methods);
    syncChips();
    render();
  }
  document.getElementById("btn-front").addEventListener("click",
    () => setActive(computeFront(metricByKey[metricKey]).methods));
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
    metricSelect.addEventListener("change", ev => {
      metricKey = ev.target.value;
      render();
    });
  }

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
    box2.innerHTML = html;
  }

  // ---------- data table ----------
  function buildTable() {
    const m0 = metricByKey[METRICS[0].key];
    const rows = [...POINTS].sort((a, b) =>
      m0.lowerBetter ? mval(a, m0) - mval(b, m0) : mval(b, m0) - mval(a, m0));
    let html = "<table><thead><tr><th>Method</th>";
    html += TRAJECTORY ? "<th>Configs</th>" : "<th>Variant</th>";
    html += "<th>Family</th>";
    for (const m of METRICS) html += `<th>${m.label}</th>`;
    html += `<th>${CONFIG.x_short}</th><th>Imputed</th></tr></thead><tbody>`;
    for (const p of rows) {
      html += `<tr><td>${p.method}</td><td>${TRAJECTORY ? (p.n_configs != null ? p.n_configs : "—") : p.variant}</td><td>${p.family}</td>`;
      for (const m of METRICS) html += `<td>${fmtMetric(m, mval(p, m))}</td>`;
      html += `<td>${p.x.toFixed(3)}</td><td>${p.imputed ? p.imputed_pct.toFixed(0) + "%" : "—"}</td></tr>`;
    }
    html += "</tbody></table>";
    document.getElementById("tblwrap").innerHTML = html;
  }

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
