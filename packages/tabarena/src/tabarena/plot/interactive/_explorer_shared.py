"""Pieces shared by every self-contained interactive explorer page.

The explorers (:mod:`tabarena.plot.interactive.pareto_explorer`,
:mod:`tabarena.plot.interactive.leaderboard_explorer`) are single dependency-free
HTML files built by substituting placeholders into a template string. Everything
that must look and behave the same across them — the color tokens, the control /
chip / tooltip / table chrome, and the small JS helper set — lives here so the
pages cannot drift apart.

Each template carries the placeholders ``__BASE_CSS__``, ``__BASE_JS__``,
``__PAGE_TITLE__``, ``__CONFIG_JSON__`` and ``__POINTS_JSON__``;
:func:`render_explorer_html` fills them in.
"""

from __future__ import annotations

import html
import json
from typing import TYPE_CHECKING

from tabarena.plot.plot_pareto_focus import FAMILY_COLORS

if TYPE_CHECKING:
    import pandas as pd

#: Family display name -> the CSS custom property carrying its color.
_FAMILY_CSS_TOKENS: dict[str, str] = {
    "Foundation Model": "--fam-foundation",
    "Neural Network": "--fam-nn",
    "Tree-based": "--fam-tree",
    "System": "--fam-system",
    "Baseline": "--fam-baseline",
    "Other": "--fam-other",
}


def _family_css_vars() -> str:
    """CSS custom-property lines carrying the shared family colors."""
    return "\n".join(f"    {token}: {FAMILY_COLORS[family]};" for family, token in _FAMILY_CSS_TOKENS.items())


#: Theme tokens + the chrome (controls, chips, tooltip, data table) every
#: explorer shares. Chart-specific layout stays in the individual templates.
#: Theme tokens, declared once per mode and emitted into four scopes below.
#: Family colors are mode-independent, so they live only in the base scope.
_LIGHT_TOKENS = """
    --paper: #ffffff;
    --card: #ffffff;
    --ink: #14161a;
    --muted: #6d6c65;
    --line: #e4e3db;
    --accent: #2a78d6;
    --chip-bg: #f2f1ec;
    --pt-muted: #b9b8b1;
    /* The same family hues as *text*. The mark colors are tuned for fills and
       drop below readable contrast as small labels on the light surface, so
       light mode darkens them; dark mode reuses the mark colors as-is. */
    --fam-foundation-ink: #7d3fc2;
    --fam-nn-ink: #1c6fa8;
    --fam-tree-ink: #2f7d32;
    --fam-system-ink: #a4600f;
    --fam-baseline-ink: #5f5f5f;
    --fam-other-ink: #5f5f5f;
    /* Tuning-variant series (default / tuned / tuned + ensembled). Light mode
       is the paper view's surface, so it uses the *static figures' own* seaborn
       pastels — figures exported from here drop straight into a paper beside
       them. The cost is colorblind separation: green vs. orange is 4.0 deutan
       ΔE, well inside the band that needs secondary encoding, which here is the
       fixed concentric bar widths plus the legend and the data table. Dark mode
       (the website) keeps the stepped, better-separated version below. */
    --var-default: #a1c9f4;
    --var-tuned: #ffb482;
    --var-tunedens: #8de5a1;
    --optimal: #228b22;
    --tooltip-bg: #14161a;
    --tooltip-ink: #fbfbf9;
    color-scheme: light;
"""

_DARK_TOKENS = """
    --paper: #131316;
    --card: #1b1b1f;
    --ink: #f0efea;
    --muted: #9b9a92;
    --line: #2e2e33;
    --accent: #3987e5;
    --chip-bg: #232327;
    --pt-muted: #55555c;
    --fam-foundation-ink: var(--fam-foundation);
    --fam-nn-ink: var(--fam-nn);
    --fam-tree-ink: var(--fam-tree);
    --fam-system-ink: var(--fam-system);
    --fam-baseline-ink: var(--fam-baseline);
    --fam-other-ink: var(--fam-other);
    --var-default: #4386d5;
    --var-tuned: #c05f38;
    --var-tunedens: #289972;
    --optimal: #2ea043;
    --tooltip-bg: #f0efea;
    --tooltip-ink: #14161a;
    color-scheme: dark;
"""


def _scope(selector: str, *bodies: str, indent: str = "  ") -> str:
    """One CSS rule, with every body line re-indented under ``selector``."""
    lines = [f"{indent}{selector} {{"]
    for body in bodies:
        lines += [f"{indent}  {line.strip()}" if line.strip() else "" for line in body.strip("\n").split("\n")]
    lines.append(f"{indent}}}")
    return "\n".join(lines)


#: Theme tokens + the chrome (controls, chips, tooltip, data table) every
#: explorer shares. Chart-specific layout stays in the individual templates.
#:
#: Four scopes, in this order: the base (light) scope carries the family colors;
#: the media query follows the OS preference; ``[data-theme="dark"]`` lets an
#: embedding page force dark (the always-dark leaderboard Space does); and
#: ``[data-theme="light"]`` forces light for the paper view, beating both the
#: media query (lower specificity) and the dark stamp (same specificity, later).
EXPLORER_BASE_CSS = "".join(
    [
        _scope(":root", "__FAMILY_CSS_VARS__", _LIGHT_TOKENS),
        "\n  @media (prefers-color-scheme: dark) {\n",
        _scope(":root", _DARK_TOKENS, indent="    "),
        "\n  }\n",
        _scope(':root[data-theme="dark"]', _DARK_TOKENS),
        "\n",
        _scope(':root[data-theme="light"]', _LIGHT_TOKENS),
        r"""
  html, body { margin: 0; background: var(--paper); }
  /* The colour emoji fonts are named *before* the generic `sans-serif`. A generic
     family matches every character through the browser's own fallback chain, so
     anything listed after it is unreachable — and that fallback resolves emoji to
     a monochrome font on Linux, which flattens the family symbols on the chips.
     Latin glyphs are unaffected: the emoji fonts carry none. */
  body {
    color: var(--ink);
    font-family: system-ui, -apple-system, "Segoe UI",
      "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
    line-height: 1.5;
    padding: 10px 12px 14px;
  }
  /* The [hidden] attribute must beat author display rules (e.g. the
     inline-flex on .metricpick), else hidden controls render empty. */
  [hidden] { display: none !important; }

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

  .chips { display: flex; flex-direction: column; gap: 9px; }
  /* One block per family: the family toggle on top, its chips wrapping below. */
  .chiprow { display: flex; flex-direction: column; align-items: flex-start; gap: 5px; }
  .famchip {
    display: inline-flex; align-items: center; gap: 6px;
    font: 650 10.5px/1.3 system-ui, sans-serif; letter-spacing: 0.06em; text-transform: uppercase;
    color: var(--muted); background: var(--chip-bg); border: 1px dashed var(--line);
    border-radius: 999px; padding: 5px 10px; cursor: pointer;
  }
  .famchip .dot { width: 8px; height: 8px; border-radius: 50%; background: var(--fam); flex: none; }
  .famchip .count { font-weight: 500; letter-spacing: 0; opacity: 0.75; }
  /* The family symbol sits at text size, not the chip's small-caps size. */
  .famchip .sym { font-size: 1.05em; letter-spacing: 0; }
  .famchip:hover { border-color: var(--fam); color: var(--ink); }
  .famchip[aria-pressed="true"] {
    border: 1px solid var(--fam);
    background: color-mix(in srgb, var(--fam) 13%, transparent);
    color: var(--ink);
  }
  /* A toggle button that carries its own colour: off is faded with a neutral
     border, on takes the colour as border and tint. Opt-in via `.toggle` so the
     older explorers, which fade their variant buttons with inline styles, are
     unaffected. */
  .btn.toggle[aria-pressed] { opacity: 0.5; }
  .btn.toggle[aria-pressed="true"] {
    opacity: 1;
    border-color: var(--fam);
    background: color-mix(in srgb, var(--fam) 18%, var(--chip-bg));
  }
  .btn.toggle .swatch {
    display: inline-block; width: 8px; height: 8px; border-radius: 50%;
    background: var(--fam); margin-right: 6px; vertical-align: middle;
  }
  .btn.toggle[aria-pressed="false"] .swatch { background: var(--pt-muted); }
  .grouplabel { font-size: 12.5px; font-weight: 600; color: var(--muted); }

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

  svg text {
    font-family: system-ui, -apple-system, "Segoe UI",
      "Apple Color Emoji", "Segoe UI Emoji", "Noto Color Emoji", sans-serif;
  }

  @media (prefers-reduced-motion: no-preference) {
    .chip, .btn, .famchip { transition: border-color 120ms ease, background-color 120ms ease; }
  }

  /* --- Paper view -----------------------------------------------------------
     A figure-ready state for slides and papers: white surface, and only the
     things needed to read the plot — the caption, the legend and the chart.
     The controls, the chip list and the data table are interactive scaffolding,
     not part of the figure. Entered via the "Paper view" button (which stamps
     data-theme="light" on the root, see the scopes above). */
  /* One toggle, in the same place in both states and never hidden — an exit
     tucked into a corner of the figure was easy to miss. */
  .viewbar { display: flex; align-items: center; gap: 10px; margin: 0 0 9px; }
  body.paper .controls,
  body.paper .chips,
  body.paper .sidebox,
  body.paper details.datatable { display: none !important; }
  body.paper { padding: 14px 18px 18px; }
  /* Export controls, revealed with the paper view. */
  .exportbar { display: flex; align-items: center; gap: 8px; margin: 0 0 10px; }
  .exportbar .hint { font-size: 12.5px; font-weight: 600; color: var(--muted); }
""",
    ]
)

#: JS every explorer needs, spliced inside each page's IIFE. Declarations plus
#: one normalization pass over ``POINTS`` (which every template defines just
#: above the splice point), so a template can use both before its own setup runs.
EXPLORER_BASE_JS = r"""
  const NS = "http://www.w3.org/2000/svg";
  // Baseline and Other are one bucket, as in the site's own type legend: they
  // already share a color, and each holds only a handful of methods.
  const FAM_MERGED = "Baseline / Other";
  const famOf = (family) => (family === "Baseline" || family === "Other" ? FAM_MERGED : family);
  // Normalized up front so every later lookup — colors, chips, sorting — sees
  // the merged family. Both templates declare POINTS above this block.
  for (const p of POINTS) p.family = famOf(p.family);

  const FAM_ORDER = ["Foundation Model", "Tree-based", "Neural Network", "System", FAM_MERGED];
  const FAM_VAR = {
    "Foundation Model": "var(--fam-foundation)",
    "Tree-based": "var(--fam-tree)",
    "Neural Network": "var(--fam-nn)",
    "System": "var(--fam-system)",
    [FAM_MERGED]: "var(--fam-baseline)",
  };
  // The symbol the website shows for each family, so a family chip here reads the
  // same as the Type column on the site. Baseline and Other are one bucket, so
  // that chip carries both symbols.
  const FAM_SYMBOL = {
    "Foundation Model": "🧠⚡",
    "Tree-based": "🌳",
    "Neural Network": "🧠🔁",
    "System": "📊",
    [FAM_MERGED]: "📏 ❓",
  };
  // Tuning-variant colours, matching the --var-* tokens the charts plot with.
  const VARIANT_VAR = {
    "Default": "var(--var-default)",
    "Tuned": "var(--var-tuned)",
    "Tuned + Ens.": "var(--var-tunedens)",
  };

  // A family chip's label: its symbol, its name and how many methods it holds.
  function famChipLabel(family, count) {
    const symbol = FAM_SYMBOL[family];
    return '<span class="dot"></span>' + (symbol ? '<span class="sym">' + symbol + "</span> " : "") +
      family + ' <span class="count">&times;' + count + "</span>";
  }

  // The same hues stepped for use as text (see the --fam-*-ink tokens).
  const FAM_INK = {
    "Foundation Model": "var(--fam-foundation-ink)",
    "Tree-based": "var(--fam-tree-ink)",
    "Neural Network": "var(--fam-nn-ink)",
    "System": "var(--fam-system-ink)",
    [FAM_MERGED]: "var(--fam-baseline-ink)",
  };

  // Create an SVG element with attributes, optionally appended to `parent`.
  function el(name, attrs, parent) {
    const node = document.createElementNS(NS, name);
    for (const k in attrs) node.setAttribute(k, attrs[k]);
    if (parent) parent.appendChild(node);
    return node;
  }

  // Rendered width of a string, measured in the live document: font metrics are
  // not knowable up front, and the label layouts need real widths to decide
  // where a name breaks and on which side of a point it fits. Memoized per
  // (weight, size), since the layouts measure the same words repeatedly.
  function makeTextMeasurer(svg, { size = 13, weight = 400 } = {}) {
    const cache = new Map();
    return function textWidth(text) {
      let w = cache.get(text);
      if (w === undefined) {
        const probe = el("text", {
          "font-size": size, "font-weight": weight, visibility: "hidden",
        }, svg);
        probe.textContent = text;
        w = probe.getComputedTextLength();
        probe.remove();
        cache.set(text, w);
      }
      return w;
    };
  }

  // Plain, ungrouped numbers with a "." decimal separator. `toFixed` is
  // locale-independent by definition, which is the point: `toLocaleString`
  // would follow the *viewer's* browser locale and print 1234,5 for a German
  // visitor, disagreeing with the figures and CSVs beside it.
  function fmtNum(v, decimals) {
    if (v == null || !isFinite(v)) return "—";
    return v.toFixed(decimals);
  }

  function fmtMetric(metric, v) {
    if (v == null || !isFinite(v)) return "—";
    return fmtNum(v, metric.decimals) + (metric.suffix || "");
  }

  function fmtTime(v) {
    if (v >= 100) return fmtNum(v, 0) + " s";
    if (v >= 1) return fmtNum(v, 1) + " s";
    if (v >= 0.1) return fmtNum(v, 2) + " s";
    return fmtNum(v, 3) + " s";
  }

  // Smallest "nice" (1/2/2.5/5 x a power of ten) step that is at least `raw`.
  function niceStep(raw) {
    if (!(raw > 0)) return 1;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    for (const m of [1, 2, 2.5, 5, 10]) {
      if (mag * m >= raw) return mag * m;
    }
    return mag * 10;
  }

  // ~`target` evenly spaced "nice" tick values covering [min, max]. Stepped by
  // index rather than by accumulation so fractional steps do not drift.
  function ticks(min, max, target) {
    const step = niceStep((max - min) / target);
    const first = Math.ceil(min / step);
    const out = [];
    for (let i = 0; first * step + i * step <= max + 1e-9; i++) out.push((first + i) * step);
    return out.length ? out : [min];
  }

  // A hover tooltip anchored inside `box` (which must be position:relative).
  function makeTooltip(box) {
    const node = box.querySelector(".tooltip");
    return {
      show(html, ev) { node.innerHTML = html; node.style.display = "block"; this.move(ev); },
      move(ev) {
        const r = box.getBoundingClientRect();
        let tx = ev.clientX - r.left + 14;
        const ty = ev.clientY - r.top + 12;
        if (tx > r.width - 270) tx = ev.clientX - r.left - 274;
        node.style.left = tx + "px";
        node.style.top = ty + "px";
      },
      hide() { node.style.display = "none"; },
    };
  }

  // Paper view — white surface, legend + chart only — is the *default*: what a
  // reader wants first is the figure, and it is the state worth exporting. The
  // controls, chip list and data table are one click away behind "Edit view".
  // `afterToggle` re-renders charts whose size is measured from the layout.
  // `options.openInPaper` (default true) decides the state the page opens in. A
  // chart opens as the figure; the leaderboard table opens with its controls,
  // since there the interaction is the point rather than scaffolding around it.
  function setUpPaperView(afterToggle, options) {
    const opts = options || {};
    const root = document.documentElement;
    let hostTheme = null;   // the embedding page's choice, captured on entry
    const btn = document.getElementById("btn-paper");
    const embedded = window.parent !== window;

    function setPaper(on) {
      document.body.classList.toggle("paper", on);
      if (on) {
        hostTheme = root.getAttribute("data-theme");
        root.setAttribute("data-theme", "light");
      } else if (hostTheme) {
        root.setAttribute("data-theme", hostTheme);
      } else {
        root.removeAttribute("data-theme");
      }
      btn.textContent = on ? "Edit view" : "Paper view";
      document.getElementById("exportbar").hidden = !on || embedded;
      if (afterToggle) requestAnimationFrame(afterToggle);
      postHeight();
    }
    btn.addEventListener("click", () => setPaper(!document.body.classList.contains("paper")));
    // Embedded, the host page owns these controls — they sit beside the panel's
    // static-figure toggle and are driven from the outside. Standalone (the
    // shareable single file) this page needs its own.
    if (embedded) document.querySelector(".viewbar").hidden = true;
    window.addEventListener("message", ev => {
      const d = ev.data;
      if (d && d.type === "tabarena-explorer-paper" && typeof d.on === "boolean") setPaper(d.on);
    });
    // Only standalone: embedded, the host owns the button and would not see the
    // key press, so its label would fall out of step with the frame.
    if (!embedded) {
      document.addEventListener("keydown", ev => {
        if (ev.key === "Escape" && !document.body.classList.contains("paper")) setPaper(true);
      });
    }
    setPaper(opts.openInPaper !== false);   // for a chart, the figure is what opens
  }

  // --- Figure export ---------------------------------------------------------
  // The chart is live SVG, so a file can be built from it directly. Three things
  // a copy has to fix up: the colors are CSS custom properties (var(--x) means
  // nothing outside this document), it has no background or font of its own, and
  // the legend is HTML rather than part of the SVG.

  // Rebuild the HTML legend as SVG, reusing its live layout: each item's glyph is
  // cloned and its label re-emitted at the measured position. foreignObject would
  // be far simpler, but Chrome refuses to rasterize it onto a canvas, which would
  // break the PNG path.
  // Rewrite every var(--x) in a clone's paint attributes; they resolve to nothing
  // once the node leaves this document.
  function resolveVars(root, resolve) {
    for (const node of [root, ...root.querySelectorAll("*")]) {
      for (const attr of ["fill", "stroke"]) {
        const value = node.getAttribute(attr);
        if (value && value.includes("var(")) node.setAttribute(attr, resolve(value));
      }
    }
  }

  function legendToSvg(container, resolve) {
    const base = container.getBoundingClientRect();
    const group = document.createElementNS(NS, "g");
    let height = 0;
    for (const item of container.querySelectorAll(".item")) {
      const box = item.getBoundingClientRect();
      if (!box.width) continue;
      height = Math.max(height, box.bottom - base.top);
      let textLeft = box.left - base.left;
      const glyph = item.querySelector("svg");
      if (glyph) {
        const gbox = glyph.getBoundingClientRect();
        const wrap = el("g", {
          transform: `translate(${gbox.left - base.left} ${gbox.top - base.top})`,
        }, group);
        const glyphClone = glyph.cloneNode(true);
        resolveVars(glyphClone, resolve);
        wrap.appendChild(glyphClone);
        textLeft = gbox.right - base.left + 5;
      }
      const label = item.textContent.trim();
      if (!label) continue;
      const colored = item.querySelector("[style*='color']");
      const text = el("text", {
        x: textLeft, y: box.top - base.top + box.height / 2 + 4, "font-size": 12.5,
        fill: resolve(getComputedStyle(colored || item).color),
      }, group);
      text.textContent = label;
    }
    return { group, height: Math.ceil(height) };
  }

  // `parts` is a list of {svg, dx}, so a chart split across panes (the sticky
  // y-axis beside the scrolling plot) still exports as one figure.
  function buildExportSvg(parts, legendEl, pad = 10) {
    const rootStyle = getComputedStyle(document.documentElement);
    const resolve = value => String(value).replace(
      /var\((--[\w-]+)\)/g, (_, name) => rootStyle.getPropertyValue(name).trim() || "none");
    const paper = rootStyle.getPropertyValue("--paper").trim() || "#ffffff";

    let chartW = 0, chartH = 0;
    for (const part of parts) {
      chartW = Math.max(chartW, part.dx + Number(part.svg.getAttribute("width")));
      chartH = Math.max(chartH, Number(part.svg.getAttribute("height")));
    }

    const out = document.createElementNS(NS, "svg");
    out.setAttribute("xmlns", NS);
    out.setAttribute("font-family", 'system-ui, -apple-system, "Segoe UI", sans-serif');
    let top = pad;
    const later = [];   // built after the width is known
    const legend = legendEl ? legendToSvg(legendEl, resolve) : null;
    if (legend && legend.height) {
      legend.group.setAttribute("transform", `translate(${pad} ${top})`);
      later.push(() => out.appendChild(legend.group));
      top += legend.height + 8;
    }

    const width = Math.max(chartW, legendEl ? legendEl.getBoundingClientRect().width : 0) + pad * 2;
    const height = top + chartH + pad;
    out.setAttribute("width", Math.ceil(width));
    out.setAttribute("height", Math.ceil(height));
    el("rect", { x: 0, y: 0, width: Math.ceil(width), height: Math.ceil(height), fill: paper }, out);
    for (const build of later) build();

    for (const part of parts) {
      const group = el("g", { transform: `translate(${part.dx + pad} ${top})` }, out);
      const clone = part.svg.cloneNode(true);
      resolveVars(clone, resolve);
      while (clone.firstChild) group.appendChild(clone.firstChild);
    }
    return out;
  }

  // Page title -> a safe file stem, e.g. "tabarena-leaderboard-explorer-all-tasks".
  function slugify(text) {
    return (text || "chart").toLowerCase().replace(/[^a-z0-9]+/g, "-").replace(/^-+|-+$/g, "").slice(0, 80);
  }

  function downloadUrl(url, filename) {
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    a.remove();
  }

  // Rasterize the export SVG into a canvas at `scale`, then hand it to `done`.
  function rasterize(svg, scale, done, fail) {
    const width = Number(svg.getAttribute("width")), height = Number(svg.getAttribute("height"));
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = Math.round(width * scale);
      canvas.height = Math.round(height * scale);
      const ctx = canvas.getContext("2d");
      ctx.setTransform(scale, 0, 0, scale, 0, 0);
      ctx.drawImage(img, 0, 0);
      done(canvas, width, height);
    };
    img.onerror = fail;
    img.src = "data:image/svg+xml;charset=utf-8,"
      + encodeURIComponent(new XMLSerializer().serializeToString(svg));
  }

  // A one-page PDF wrapping the rendered figure, written by hand: a library would
  // cost this page its zero-dependency, single-file property. The image is stored
  // losslessly (raw RGB + /FlateDecode via CompressionStream) and the page is sized
  // in points to the figure's CSS size, so it prints at the size it appears here
  // and the pixels land at 96*scale dpi.
  async function buildPdf(canvas, cssWidth, cssHeight) {
    const pixels = canvas.getContext("2d").getImageData(0, 0, canvas.width, canvas.height).data;
    const rgb = new Uint8Array((pixels.length / 4) * 3);
    for (let i = 0, j = 0; i < pixels.length; i += 4, j += 3) {
      rgb[j] = pixels[i];
      rgb[j + 1] = pixels[i + 1];
      rgb[j + 2] = pixels[i + 2];
    }
    const deflated = new Uint8Array(await new Response(
      new Blob([rgb]).stream().pipeThrough(new CompressionStream("deflate"))).arrayBuffer());

    const encoder = new TextEncoder();
    const chunks = [];
    const offsets = [];
    let cursor = 0;
    const put = data => {
      const bytes = typeof data === "string" ? encoder.encode(data) : data;
      chunks.push(bytes);
      cursor += bytes.length;
    };
    const object = (id, body, stream) => {
      offsets[id] = cursor;
      put(`${id} 0 obj\n${body}\n`);
      if (stream) {
        put("stream\n");
        put(stream);
        put("\nendstream\n");
      }
      put("endobj\n");
    };

    const ptW = (cssWidth * 0.75).toFixed(2), ptH = (cssHeight * 0.75).toFixed(2);
    const content = `q ${ptW} 0 0 ${ptH} 0 0 cm /Im0 Do Q`;
    put("%PDF-1.4\n");
    put(new Uint8Array([0x25, 0xe2, 0xe3, 0xcf, 0xd3, 0x0a]));   // binary marker
    object(1, "<< /Type /Catalog /Pages 2 0 R >>");
    object(2, "<< /Type /Pages /Kids [3 0 R] /Count 1 >>");
    object(3, `<< /Type /Page /Parent 2 0 R /MediaBox [0 0 ${ptW} ${ptH}] `
      + "/Resources << /XObject << /Im0 4 0 R >> >> /Contents 5 0 R >>");
    object(4, "<< /Type /XObject /Subtype /Image "
      + `/Width ${canvas.width} /Height ${canvas.height} /ColorSpace /DeviceRGB `
      + `/BitsPerComponent 8 /Filter /FlateDecode /Length ${deflated.length} >>`, deflated);
    object(5, `<< /Length ${content.length} >>`, content);

    const xref = cursor;
    let table = "xref\n0 6\n0000000000 65535 f \n";
    for (let id = 1; id <= 5; id++) table += String(offsets[id]).padStart(10, "0") + " 00000 n \n";
    put(table);
    put(`trailer\n<< /Size 6 /Root 1 0 R >>\nstartxref\n${xref}\n%%EOF\n`);
    return new Blob(chunks, { type: "application/pdf" });
  }

  // Wire up the export buttons; `getParts` is called per click so the file always
  // matches what is on screen. Returns a `run(format)` the host page can drive.
  function setUpExport(getParts, basename) {
    const buttons = {
      svg: document.getElementById("btn-svg"),
      png: document.getElementById("btn-png"),
      pdf: document.getElementById("btn-pdf"),
    };
    const figure = () => buildExportSvg(getParts(), document.getElementById("legendstrip"));

    // A sandboxed frame has no modals, so a failure is reported on the button.
    function complain(format) {
      const button = buttons[format];
      if (!button) return;
      const label = button.textContent;
      button.textContent = "failed";
      setTimeout(() => { button.textContent = label; }, 2500);
    }

    function run(format) {
      const svg = figure();
      const name = basename();
      if (format === "svg") {
        downloadUrl("data:image/svg+xml;charset=utf-8,"
          + encodeURIComponent(new XMLSerializer().serializeToString(svg)), name + ".svg");
        return;
      }
      // 3x for a screen-resolution PNG; 2x for the PDF, whose page is sized in
      // points so the pixels already land near 200 dpi at print size.
      rasterize(svg, format === "pdf" ? 2 : 3, (canvas, cssWidth, cssHeight) => {
        if (format === "png") {
          canvas.toBlob(blob => downloadUrl(URL.createObjectURL(blob), name + ".png"), "image/png");
        } else {
          buildPdf(canvas, cssWidth, cssHeight)
            .then(blob => downloadUrl(URL.createObjectURL(blob), name + ".pdf"))
            .catch(() => complain("pdf"));
        }
      }, () => complain(format));
    }

    for (const format of Object.keys(buttons)) {
      if (buttons[format]) buttons[format].addEventListener("click", () => run(format));
    }
    // Embedded, the buttons live in the host's panel header (see main.taExport).
    window.addEventListener("message", ev => {
      const d = ev.data;
      if (d && d.type === "tabarena-explorer-export" && buttons[d.format] !== undefined) run(d.format);
    });
  }

  // When embedded, report the content height so the host page can size the
  // iframe to fit (avoids an inner scrollbar). Works from a sandboxed frame.
  // Measure the body (viewport-independent) — documentElement.scrollHeight is
  // clamped to at least the iframe's current viewport, which turns the
  // resize round-trip into a grow-forever feedback loop. The change guard
  // stops re-posting once the height settles.
  let lastPostedHeight = 0;
  function postHeight() {
    if (window.parent === window) return;
    const height = Math.ceil(document.body.offsetHeight);
    if (Math.abs(height - lastPostedHeight) < 3) return;
    lastPostedHeight = height;
    window.parent.postMessage({ type: "tabarena-explorer-height", height: height }, "*");
  }
"""


def render_explorer_html(
    template: str,
    *,
    page_title: str,
    config: dict,
    points: pd.DataFrame,
) -> str:
    """Fill an explorer template's placeholders and return the finished page."""
    return (
        template.replace("__BASE_CSS__", EXPLORER_BASE_CSS)
        .replace("__BASE_JS__", EXPLORER_BASE_JS)
        .replace("__FAMILY_CSS_VARS__", _family_css_vars())
        .replace("__PAGE_TITLE__", html.escape(page_title))
        .replace("__CONFIG_JSON__", json.dumps(config))
        .replace("__POINTS_JSON__", points.to_json(orient="records"))
    )
