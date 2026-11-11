"""Build the self-contained interactive composite leaderboard HTML.

The interactive twin of ``composite_leaderboard.png``: built from the frame
:func:`tabarena.plot.composite_leaderboard.build_composite` returns (one row per method and metric,
one column per evaluation subset), so the page, the CSV and the PNG show the same numbers. The page
adds what a picture cannot: sorting by any subset, one metric at a time or both, the families to
show, a search, and the reader's own picks of methods to mark or take out. Each subset's column is
colour-graded over the methods on screen, as the PNG grades it over its rows.
"""

from __future__ import annotations

from collections.abc import Container
from pathlib import Path

import pandas as pd

from tabarena.plot.interactive._composite_template import COMPOSITE_TEMPLATE
from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive.winrate_explorer import _split_label
from tabarena.website.website_format import get_model_family


def build_composite_explorer_html(
    composite: pd.DataFrame,
    *,
    save_path: str | Path,
    higher_better: dict[str, bool],
    title: str | None = None,
    page_title: str = "TabArena composite leaderboard",
    system_names: Container[str] = frozenset(),
) -> Path | None:
    """Render the interactive composite leaderboard.

    Parameters
    ----------
    composite
        The composite frame: a ``(method, metric)`` row index, one column per subset, as
        :func:`tabarena.plot.composite_leaderboard.build_composite` returns it (methods in its
        order, which the page opens on).
    save_path
        Where to write the page.
    higher_better
        Metric display name -> whether a higher value is better (Elo) or a lower one (improvability);
        sets the direction of each column's grading and of a column's default sort.
    title
        Headline above the table; none when ``None``.
    system_names
        Display names of the systems, which are never classified from their name.

    Returns:
    -------
    The written path, or ``None`` when the composite is empty.
    """
    if composite.empty:
        return None
    methods = [str(m) for m in dict.fromkeys(composite.index.get_level_values("method"))]
    metrics = [str(m) for m in dict.fromkeys(composite.index.get_level_values("metric"))]
    subsets = [str(c) for c in composite.columns]
    values = {
        metric: [
            [None if pd.isna(v) else float(v) for v in composite.loc[(method, metric)].to_numpy()]
            if (method, metric) in composite.index
            else [None] * len(subsets)
            for method in methods
        ]
        for metric in metrics
    }
    split = [_split_label(m) for m in methods]
    points = pd.DataFrame(
        {
            "method": methods,
            "model": [model for model, _ in split],
            "variant": [variant for _, variant in split],
            "family": [get_model_family(model, system_names=system_names) for model, _ in split],
        }
    )
    config = {
        "title": title,
        "subsets": subsets,
        "metrics": metrics,
        "values": values,
        "higherBetter": {metric: bool(higher_better.get(metric, True)) for metric in metrics},
    }
    html = render_explorer_html(COMPOSITE_TEMPLATE, page_title=page_title, config=config, points=points)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
