"""Build the self-contained interactive win-rate matrix HTML.

The interactive twin of the static ``winrate_matrix`` figure. Built from the same
square matrix :meth:`bencheval.TabArena.compute_winrate_matrix` returns, so the
two cannot disagree, and written beside the figure so the website can offer the
same static / interactive / paper triple it offers for every other plot.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._winrate_template import WINRATE_TEMPLATE
from tabarena.plot.interactive.leaderboard_explorer import _VARIANT_LABELS
from tabarena.website.website_format import get_model_family

#: A trailing tuning-variant tag on a matrix label, e.g. "RealMLP (tuned)". A
#: parenthetical that is not a tuning variant (the reference pipeline "AutoGluon
#: 1.5 (extreme, 4h)") stays part of the model name.
_VARIANT_TAG_RE = re.compile(r"\s*\((default|tuned \+ ensembled|tuned)\)\s*$")


def _split_label(label: str) -> tuple[str, str]:
    """A matrix label -> (model, variant), the variant spelled as the charts spell it."""
    match = _VARIANT_TAG_RE.search(label)
    if not match:
        return label, ""
    return _VARIANT_TAG_RE.sub("", label), _VARIANT_LABELS[match.group(1)]


def _families(labels: list[str]) -> dict[str, str]:
    """Matrix label -> model family, best effort.

    Labels are display names (the matrix is built after the reporter's renames),
    which is one of the two forms :func:`get_model_family` accepts. Unknown names
    fall through to its own default rather than raising, so a new model missing
    from the family table still renders — in the neutral bucket.
    """
    return {label: get_model_family(_split_label(label)[0]) for label in labels}


def build_winrate_explorer_html(
    winrate_matrix: pd.DataFrame,
    *,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena win-rate matrix",
) -> Path | None:
    """Render the interactive win-rate matrix.

    Parameters
    ----------
    winrate_matrix
        Square frame indexed and columned by method, entry (i, j) the win rate of
        method i over method j, as returned by
        :meth:`bencheval.TabArena.compute_winrate_matrix`.
    save_path
        Where to write the page.
    title
        Headline shown above the matrix; omitted when ``None``.

    Returns:
    -------
    The written path, or ``None`` when there is nothing to compare (fewer than two
    methods).
    """
    if winrate_matrix.shape[0] < 2:
        return None
    # Align the columns to the row order, so the matrix the page indexes row-major
    # is the same matrix in both directions.
    methods = [str(m) for m in winrate_matrix.index]
    matrix = winrate_matrix.reindex(index=winrate_matrix.index, columns=winrate_matrix.index)
    values = matrix.astype(float).where(pd.notna(matrix), None).to_numpy().tolist()

    families = _families(methods)
    # The mean excludes the diagonal (a method against itself is not a comparison).
    numeric = matrix.astype(float)
    means = [
        float(numeric.iloc[i].drop(numeric.index[i]).mean(skipna=True))
        for i in range(len(methods))
    ]
    split = [_split_label(m) for m in methods]
    points = pd.DataFrame(
        {
            "method": methods,
            "model": [model for model, _ in split],
            "variant": [variant for _, variant in split],
            "family": [families[m] for m in methods],
            "mean": means,
        }
    )

    config = {
        "title": title,
        "methods": methods,
        "matrix": values,
        "variants": list(_VARIANT_LABELS.values()),
    }
    html = render_explorer_html(
        WINRATE_TEMPLATE, page_title=page_title, config=config, points=points
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
