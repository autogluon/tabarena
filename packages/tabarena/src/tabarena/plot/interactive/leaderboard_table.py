"""Build the self-contained interactive full leaderboard table HTML.

The interactive twin of the website's leaderboard table. Like
:mod:`tabarena.plot.interactive.leaderboard_explorer` it is built from the
*website* table (``website_leaderboard.csv``), so the table the reader sorts can
never disagree with the numbers beside it, and refreshing it is part of the fast
artifact-conversion step rather than a re-evaluation.

It lives here rather than in the leaderboard Space so that it reuses the
explorers' family and variant colors, chip components and imputation markers
instead of reimplementing them in the Space's Gradio CSS.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._leaderboard_table_template import LEADERBOARD_TABLE_TEMPLATE
from tabarena.plot.interactive.leaderboard_explorer import (
    _VARIANT_LABELS,
    _parse_model,
)
from tabarena.website.website_format import Constants

#: Family display name -> the symbol the website shows in its ``Type`` column.
#: Mirrors ``constants.model_type_emoji`` in the leaderboard app.
_FAMILY_SYMBOL = {
    Constants.foundational: "🧠⚡",
    Constants.neural_network: "🧠🔁",
    Constants.tree: "🌳",
    Constants.baseline: "📏",
    Constants.reference: "📊",
    "Other": "❓",
}

#: One entry per rendered column, in display order. ``key`` is the field on the
#: point records; ``always`` columns cannot be switched off; ``heatmap`` columns
#: are shaded per column; ``logScale`` ones are shaded in log space because
#: runtimes span orders of magnitude. ``hint`` is the header's hover text — kept
#: to the direction and the unit, since the page embedding this one carries the
#: full metric definitions.
_COLUMNS: list[dict] = [
    {
        "key": "position",
        "label": "#",
        "always": True,
        "hint": "Position in this subset's published ranking.",
    },
    {
        "key": "family",
        "label": "Type",
        "always": True,
        "hint": "Model family — see the key above the table.",
    },
    {
        "key": "model",
        "label": "Model",
        "always": True,
        "hint": (
            "The model, its tuning variant in brackets, ✔️ when the implementation was "
            "verified, and ‡ when some results are imputed."
        ),
    },
    {
        "key": "elo",
        "column": "Elo [⬆️]",
        "label": "Elo",
        "decimals": 0,
        "heatmap": True,
        "hint": "Pairwise win-rate rating; a 400-point gap is about a 91% win rate. Higher is better.",
    },
    {
        "key": "score",
        "column": "Score [⬆️]",
        "label": "Score",
        "decimals": 3,
        "heatmap": True,
        "hint": "Error rescaled per dataset to 1 (best) … 0 (median), then averaged. Higher is better.",
    },
    {
        "key": "rank",
        "column": "Rank [⬇️]",
        "label": "Rank",
        "decimals": 2,
        "lowerBetter": True,
        "heatmap": True,
        "hint": "Mean rank across datasets. Lower is better.",
    },
    {
        "key": "hrank",
        "column": "Harmonic Rank [⬇️]",
        "label": "Harmonic rank",
        "short": "H. rank",
        "decimals": 2,
        "lowerBetter": True,
        "heatmap": True,
        "hint": "Harmonic mean of per-dataset ranks; rewards being excellent somewhere. Lower is better.",
    },
    {
        "key": "imp",
        "column": "Improvability (%) [⬇️]",
        "label": "Improvability (%)",
        "short": "Improv.",
        "decimals": 2,
        "lowerBetter": True,
        "heatmap": True,
        "hint": "How much lower the best model's error is than this one's, per dataset. Lower is better.",
    },
    {
        "key": "train_time",
        "column": "Median Train Time (s/1K) [⬇️]",
        "label": "Train time (s/1K)",
        "short": "Train s",
        "decimals": 2,
        "lowerBetter": True,
        "heatmap": True,
        "logScale": True,
        "hint": "Median seconds to fit per 1000 rows. Lower is better; shaded on a log scale.",
    },
    {
        "key": "predict_time",
        "column": "Median Predict Time (s/1K) [⬇️]",
        "label": "Predict time (s/1K)",
        "short": "Predict s",
        "decimals": 3,
        "lowerBetter": True,
        "heatmap": True,
        "logScale": True,
        "hint": "Median seconds to predict per 1000 rows. Lower is better; shaded on a log scale.",
    },
    {
        "key": "imputed_pct",
        "column": "Imputed (%) [⬇️]",
        "label": "Imputed (%)",
        "short": "Imputed",
        "decimals": 1,
        "lowerBetter": True,
        "heatmap": True,
        "hint": "Share of datasets whose score was imputed because the model could not run on them.",
    },
    {
        "key": "hardware",
        "column": "Hardware",
        "label": "Hardware",
        "text": True,
        "hint": "The hardware the reported runtimes were measured on.",
    },
]

#: The metric the medals and the chip ordering use.
_RANK_KEY = "elo"



def leaderboard_table_points(df_website_leaderboard: pd.DataFrame) -> pd.DataFrame:
    """One record per leaderboard row, with the fields the table renders."""
    df = df_website_leaderboard.reset_index(drop=True)
    parsed = [_parse_model(m) for m in df["Model"]]
    points = pd.DataFrame(
        {
            "position": (
                pd.to_numeric(df["#"], errors="coerce")
                if "#" in df.columns
                else pd.Series(range(len(df)))
            ),
            "method": [p[0] for p in parsed],
            "variant": [p[1] for p in parsed],
            "family": df["TypeName"].to_numpy(),
            "url": [p[2] for p in parsed],
        }
    )
    points["family_symbol"] = [_FAMILY_SYMBOL.get(f, "❓") for f in df["TypeName"]]
    points["verified"] = (
        df["Verified"].astype(str).str.strip().eq("✔️").to_numpy()
        if "Verified" in df.columns
        else False
    )
    for spec in _COLUMNS:
        column = spec.get("column")
        if not column:
            continue
        if spec.get("text"):
            points[spec["key"]] = (
                df[column].to_numpy() if column in df.columns else None
            )
        else:
            points[spec["key"]] = (
                pd.to_numeric(df[column], errors="coerce").to_numpy()
                if column in df.columns
                else pd.NA
            )
    points["elo_ci"] = df["Elo 95% CI"].to_numpy() if "Elo 95% CI" in df.columns else None

    imputed_pct = pd.to_numeric(points.get("imputed_pct"), errors="coerce").fillna(0.0)
    points["imputed_pct"] = imputed_pct.to_numpy()
    # `Imputed` is the website's own flag; fall back to the percentage when a
    # table predates the column.
    if "Imputed" in df.columns:
        points["imputed"] = df["Imputed"].astype(bool).to_numpy()
    else:
        points["imputed"] = (imputed_pct > 0).to_numpy()
    return points.dropna(subset=["elo"]).reset_index(drop=True)


def build_leaderboard_table_html(
    df_website_leaderboard: pd.DataFrame,
    *,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena leaderboard table",
) -> Path | None:
    """Render the interactive full leaderboard table for one subset's website table.

    Parameters
    ----------
    df_website_leaderboard
        A ``website_leaderboard.csv`` frame (see
        :func:`tabarena.website.website_format.format_leaderboard`). Required
        columns: ``Model``, ``TypeName`` and ``Elo [⬆️]``; every other column of
        :data:`_COLUMNS` is rendered when present.
    save_path
        Where to write the page.
    title
        Headline shown above the table; omitted when ``None``.

    Returns:
    -------
    The written path, or ``None`` when the table has no usable Elo values.
    """
    for column in ("Model", "TypeName", "Elo [⬆️]"):
        if column not in df_website_leaderboard.columns:
            raise ValueError(f"website leaderboard is missing required column {column!r}")

    points = leaderboard_table_points(df_website_leaderboard)
    if points.empty:
        return None

    columns = []
    for spec in _COLUMNS:
        source = spec.get("column")
        if source and source not in df_website_leaderboard.columns:
            continue
        if source and not spec.get("text") and not points[spec["key"]].notna().any():
            continue
        columns.append({k: v for k, v in spec.items() if k != "column"})

    config = {
        "title": title,
        "columns": columns,
        "rankKey": _RANK_KEY,
        "variants": list(_VARIANT_LABELS.values()),
    }
    html = render_explorer_html(
        LEADERBOARD_TABLE_TEMPLATE, page_title=page_title, config=config, points=points
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
