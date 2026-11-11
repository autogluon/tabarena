"""HTML/CSS/JS template for the tabbed page over a ``generate_all_figs`` run's explorers.

Kept as a Python string constant so it ships with the package without any build-system data-file
configuration; :func:`tabarena.plot.interactive.explorer_hub.build_explorer_hub_html` fills in the
title, the shared stylesheet and the manifest of pages.
"""

from __future__ import annotations

HUB_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__PAGE_TITLE__</title>
<style>
__BASE_CSS__

  body { padding: 10px 12px 16px; }
  .hub-head { display: flex; flex-wrap: wrap; align-items: center; gap: 8px 18px; margin-bottom: 6px; }
  .hub-title { margin: 0; font: 700 17px/1.3 system-ui, sans-serif; color: var(--ink); }
  .hub-subset { display: inline-flex; align-items: center; gap: 7px; }
  .hub-subset[hidden] { display: none; }
  .hub-head .btn { margin-left: auto; }
  .hub-subset select {
    font: 500 12.5px/1.2 system-ui, sans-serif; color: var(--ink); background: var(--chip-bg);
    border: 1px solid var(--line); border-radius: 7px; padding: 5px 8px; max-width: 320px;
  }
  .hub-tabs { display: flex; flex-wrap: wrap; align-items: flex-end; gap: 2px; border-bottom: 1px solid var(--line); margin-bottom: 10px; }
  .hub-tab {
    font: 600 13px/1 system-ui, sans-serif; color: var(--muted); background: none; cursor: pointer;
    border: 0; border-bottom: 2px solid transparent; padding: 9px 12px 8px; margin-bottom: -1px;
  }
  .hub-tab:hover { color: var(--ink); }
  .hub-tab[aria-selected="true"] { color: var(--ink); border-bottom-color: var(--accent); }
  .hub-tab:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  .hub-sep { align-self: stretch; width: 1px; background: var(--line); margin: 7px 8px; }
  .hub-frame { display: block; width: 100%; height: 82vh; border: 0; }
  .hub-frame[hidden] { display: none; }
</style>
</head>
<body>
  <div class="hub-head">
    <h1 class="hub-title">__PAGE_TITLE__</h1>
    <label class="hub-subset" id="subset-box" hidden><span class="grouplabel">Subset</span><select id="subset"></select></label>
    <button class="btn" id="paper" type="button" title="Switch the page shown between its controls and the figure alone">Paper view</button>
  </div>
  <div class="hub-tabs" role="tablist" id="tabs" aria-label="Explorers"></div>
  <div id="frames"></div>

<script>
(function () {
  "use strict";
  const M = __MANIFEST_JSON__;
  // One entry per tab: the run's composite pages first, then the pages each subset has.
  const TABS = [
    ...M.root.map((r, i) => ({ key: "r" + i, label: r.label, href: () => r.href })),
    ...M.subsetTabs.map(t => ({ key: t.key, label: t.label, pages: t.pages, href: s => t.pages[s] })),
  ];
  const state = { tab: TABS[0].key, subset: M.subsets[0] || null };
  const tabsBox = document.getElementById("tabs"), framesBox = document.getElementById("frames");
  const subsetBox = document.getElementById("subset-box"), subsetPick = document.getElementById("subset");
  const frames = new Map();
  // This page is the explorers' host, as the website is: inside a frame an explorer hides its own view bar and a
  // chart opens on its figure, waiting for the host to switch it. Each page is switched to its edit view (its
  // controls, as it opens standalone) once it loads, takes this page's theme, and the Paper view button switches
  // the page shown between the two.
  const paperBtn = document.getElementById("paper");
  const paperOn = new Map();
  const theme = () => document.documentElement.getAttribute("data-theme") ||
    (window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light");
  function setPaper(frame, on) {
    paperOn.set(frame, on);
    if (frame.contentWindow) frame.contentWindow.postMessage({ type: "tabarena-explorer-paper", on: on }, "*");
    syncPaperButton();
  }
  function current() { return [...frames.values()].find(f => !f.hidden) || null; }
  function syncPaperButton() {
    const f = current();
    paperBtn.textContent = f && paperOn.get(f) ? "Edit view" : "Paper view";
  }
  paperBtn.addEventListener("click", () => { const f = current(); if (f) setPaper(f, !paperOn.get(f)); });

  TABS.forEach((t, i) => {
    if (i === M.root.length && i > 0) {
      const sep = document.createElement("span");
      sep.className = "hub-sep";
      tabsBox.appendChild(sep);
    }
    const b = document.createElement("button");
    b.className = "hub-tab";
    b.type = "button";
    b.setAttribute("role", "tab");
    b.dataset.key = t.key;
    b.textContent = t.label;
    b.addEventListener("click", () => { state.tab = t.key; show(); });
    tabsBox.appendChild(b);
  });
  tabsBox.addEventListener("keydown", ev => {
    const step = { ArrowRight: 1, ArrowLeft: -1 }[ev.key];
    if (!step) return;
    const at = TABS.findIndex(t => t.key === state.tab);
    state.tab = TABS[(at + step + TABS.length) % TABS.length].key;
    show();
    tabsBox.querySelector(`[data-key="${state.tab}"]`).focus();
  });
  subsetPick.addEventListener("change", () => { state.subset = subsetPick.value; show(); });

  function show() {
    const tab = TABS.find(t => t.key === state.tab) || TABS[0];
    state.tab = tab.key;
    for (const b of tabsBox.querySelectorAll(".hub-tab")) b.setAttribute("aria-selected", String(b.dataset.key === tab.key));
    let href, frameKey;
    if (tab.pages) {
      const choices = M.subsets.filter(s => s in tab.pages);
      if (!choices.includes(state.subset)) state.subset = choices[0];
      const esc = text => String(text).replace(/[&<>"]/g, c => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
      subsetPick.innerHTML = choices.map(s => `<option value="${esc(s)}">${esc((M.labels || {})[s] || s)}</option>`).join("");
      subsetPick.value = state.subset;
      subsetBox.hidden = false;
      href = tab.href(state.subset);
      frameKey = tab.key + "|" + state.subset;
    } else {
      subsetBox.hidden = true;
      href = tab.href();
      frameKey = tab.key;
    }
    // A page is loaded the first time it is shown and kept, so coming back to it keeps its state.
    let frame = frames.get(frameKey);
    if (!frame) {
      frame = document.createElement("iframe");
      frame.className = "hub-frame";
      frame.title = tab.label + (tab.pages ? " · " + state.subset : "");
      frame.src = href;
      frame.addEventListener("load", () => {
        frame.contentWindow.postMessage({ type: "tabarena-explorer-theme", theme: theme() }, "*");
        setPaper(frame, false);
      });
      framesBox.appendChild(frame);
      frames.set(frameKey, frame);
    }
    for (const f of frames.values()) f.hidden = f !== frame;
    syncPaperButton();
    writeHash();
  }
  // Each explorer reports its height when embedded; its frame takes it, so the page scrolls as one.
  window.addEventListener("message", ev => {
    const d = ev.data;
    if (!d || d.type !== "tabarena-explorer-height" || typeof d.height !== "number") return;
    for (const f of frames.values()) if (f.contentWindow === ev.source) f.style.height = Math.max(200, d.height + 4) + "px";
  });

  function writeHash() {
    const params = new URLSearchParams({ tab: state.tab });
    const tab = TABS.find(t => t.key === state.tab);
    if (tab && tab.pages && state.subset) params.set("subset", state.subset);
    history.replaceState(null, "", "#" + params.toString());
  }
  function readHash() {
    const params = new URLSearchParams(location.hash.slice(1));
    if (TABS.some(t => t.key === params.get("tab"))) state.tab = params.get("tab");
    if (M.subsets.includes(params.get("subset"))) state.subset = params.get("subset");
  }
  readHash();
  show();
})();
</script>
</body>
</html>
"""
