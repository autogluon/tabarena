"""Build one page holding every explorer of a ``generate_all_figs`` output directory, one tab each.

The explorers stay separate files; the page loads each into a frame the first time its tab is
opened and keeps it, so switching tabs keeps a page's state. The composite pages of the whole
run (the composite leaderboard, Pareto front, tuning trajectories and diagnostics) are tabs of
their own; the pages written per subset (per dataset, head to head, win rates) are tabs too, with
a picker for the subset. The page is the explorers' host, as the website is: it sizes each frame from
the explorers' height message, opens each explorer in its edit view (with its controls) and switches
the one shown to its paper view and back from a button of its own.
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from urllib.parse import quote

from tabarena.plot.composite_leaderboard import _strip_lite, order_subsets
from tabarena.plot.interactive._explorer_shared import EXPLORER_BASE_CSS, _family_css_vars
from tabarena.plot.interactive._hub_template import HUB_TEMPLATE

#: (file at the output directory's root, tab label), in tab order.
ROOT_PAGES: tuple[tuple[str, str], ...] = (
    ("composite_leaderboard_explorer.html", "Leaderboard"),
    ("pareto_front_explorer.html", "Pareto front"),
    ("tuning_trajectories_explorer.html", "Tuning trajectories"),
    ("tuning_diagnostics_explorer.html", "Tuning diagnostics"),
)
#: (file in a subset's directory, tab key, tab label), in tab order.
SUBSET_PAGES: tuple[tuple[str, str, str], ...] = (
    ("per_dataset_explorer.html", "per_dataset", "Per dataset"),
    ("head_to_head_explorer.html", "head_to_head", "Head to head"),
    ("winrate_explorer.html", "winrate", "Win rates"),
)


def _href(*parts: str) -> str:
    """A relative link to ``parts``, each encoded as a path segment: the characters RFC 3986 allows in one
    (as the ``!`` and ``&`` of subset names like ``core&!large``) stay as they are.
    """
    return "/".join(quote(part, safe="!$&'()*+,;=:@") for part in parts)


def build_explorer_hub_html(
    output_dir: str | Path,
    *,
    subsets: list[str] | None = None,
    save_path: str | Path | None = None,
    page_title: str = "TabArena explorers",
    strip_lite: bool | None = None,
) -> Path | None:
    """Write the tabbed page over the explorers found in ``output_dir``.

    Parameters
    ----------
    output_dir
        A ``generate_all_figs`` output directory: the composite pages at its root, each subset's
        pages in ``<output_dir>/<subset>/``.
    subsets
        The subset directories (``generate_all_figs``' output suffixes); ``None`` takes every directory
        holding one of :data:`SUBSET_PAGES`. The picker names and orders them as the composite
        leaderboard names and orders its columns.
    save_path
        Where to write the page; ``<output_dir>/explorers.html`` when ``None``. Links are relative to
        it, so it has to sit in ``output_dir``.
    page_title
        The browser title and the headline.
    strip_lite
        Whether the subset picker drops the ``lite`` token from the subsets' names, as the composite
        leaderboard names its columns (:func:`tabarena.plot.composite_leaderboard.normalize_subset_names`);
        ``None`` drops it when every subset carries it.

    Returns:
    -------
    The written path, or ``None`` when ``output_dir`` holds no explorer.
    """
    output_dir = Path(output_dir)
    if subsets is None:
        subsets = sorted(
            p.name for p in output_dir.iterdir() if p.is_dir() and any((p / f).is_file() for f, _, _ in SUBSET_PAGES)
        )
    root = [{"label": label, "href": _href(f)} for f, label in ROOT_PAGES if (output_dir / f).is_file()]
    subset_tabs = []
    for f, key, label in SUBSET_PAGES:
        pages = {s: _href(s, f) for s in subsets if (output_dir / s / f).is_file()}
        if pages:
            subset_tabs.append({"key": key, "label": label, "pages": pages})
    if not root and not subset_tabs:
        return None
    kept = [s for s in subsets if any(s in t["pages"] for t in subset_tabs)]
    if strip_lite is None:
        strip_lite = bool(kept) and all("lite" in s.split("&") for s in kept)
    by_label = {_strip_lite(s) if strip_lite else s: s for s in kept}
    kept = [by_label[label] for label in order_subsets(list(by_label))]
    manifest = {
        "root": root,
        "subsetTabs": subset_tabs,
        "subsets": kept,
        # The names the composite leaderboard gives the subsets, for the picker; the links keep the directories'.
        "labels": {s: _strip_lite(s) if strip_lite else s for s in kept},
    }
    page = (
        HUB_TEMPLATE.replace("__BASE_CSS__", EXPLORER_BASE_CSS)
        .replace("__FAMILY_CSS_VARS__", _family_css_vars())
        .replace("__PAGE_TITLE__", html.escape(page_title))
        .replace("__MANIFEST_JSON__", json.dumps(manifest).replace("</", "<\\/"))
    )
    save_path = output_dir / "explorers.html" if save_path is None else Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(page, encoding="utf-8")
    return save_path
