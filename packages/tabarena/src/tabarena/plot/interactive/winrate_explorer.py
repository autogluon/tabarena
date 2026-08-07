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
#: parenthetical that is not a tuning variant (a system's "AutoGluon 1.5 (extreme, 4h)")
#: stays part of the model name. The reporter shortens the tuned + ensembled tag to
#: "(T+E)" before the matrix is computed, so that spelling has to be here too: without it
#: every tuned + ensembled row reads as variant-less and the variant toggles collapse to
#: a lone "Default".
_VARIANT_TAG_RE = re.compile(r"\s*\((default|tuned \+ ensembled?|T\+E|tuned)\)\s*$", re.IGNORECASE)

#: Tag as written in a matrix label -> key in :data:`_VARIANT_LABELS`.
_VARIANT_TAG_KEYS = {
    "default": "default",
    "tuned": "tuned",
    "tuned + ensemble": "tuned + ensembled",
    "tuned + ensembled": "tuned + ensembled",
    "t+e": "tuned + ensembled",
}


def _split_label(label: str) -> tuple[str, str]:
    """A matrix label -> (model, variant), the variant spelled as the charts spell it."""
    match = _VARIANT_TAG_RE.search(label)
    if not match:
        return label, ""
    return _VARIANT_TAG_RE.sub("", label), _VARIANT_LABELS[_VARIANT_TAG_KEYS[match.group(1).lower()]]


def _families(labels: list[str], system_names: frozenset[str] = frozenset()) -> dict[str, str]:
    """Matrix label -> model family, best effort.

    Labels are display names (the matrix is built after the reporter's renames),
    which is one of the two forms :func:`get_model_family` accepts. Unknown names
    fall through to its own default rather than raising, so a new model missing
    from the family table still renders — in the neutral bucket. ``system_names`` carries
    the display names `method_class` declares to be systems, which are never name-inferred.
    """
    return {label: get_model_family(_split_label(label)[0], system_names=system_names) for label in labels}


def build_winrate_explorer_html(
    winrate_matrix: pd.DataFrame,
    *,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena win-rate matrix",
    system_names: frozenset[str] = frozenset(),
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
    # Rounded before serializing: the page prints a win rate to one decimal of a percent, and
    # a full-variant matrix is a few thousand cells, where the unrounded floats are most of
    # the file. `None` for the NaN diagonal, which JSON has no literal for.
    values = matrix.astype(float).round(4).where(pd.notna(matrix), None).to_numpy().tolist()

    families = _families(methods, system_names=system_names)
    # The mean excludes the diagonal (a method against itself is not a comparison). Rounded
    # like the cells above, and for the same reason.
    numeric = matrix.astype(float)
    means = [round(float(numeric.iloc[i].drop(numeric.index[i]).mean(skipna=True)), 4) for i in range(len(methods))]
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
    html = render_explorer_html(WINRATE_TEMPLATE, page_title=page_title, config=config, points=points)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
