"""Build the self-contained interactive leaderboard-overview explorer HTML.

The interactive twin of the static ``tuning-impact-elo`` bar figure. It is built
from the *website* leaderboard table (``website_leaderboard.csv``) rather than
from the raw evaluation frame, so the chart and the table below it on
tabarena.ai can never disagree — and so refreshing it is part of the fast
artifact-conversion step instead of a full re-evaluation.
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._leaderboard_template import LEADERBOARD_TEMPLATE
from tabarena.website.website_format import Constants

#: ``Model`` cell -> variant. The website spells the variants out in the model
#: name; the explorer wants them as a separate series.
_VARIANT_RE = re.compile(r"\((default|tuned \+ ensembled|tuned)\)")
_VARIANT_LABELS = {"default": "Default", "tuned": "Tuned", "tuned + ensembled": "Tuned + Ens."}

#: ``+106/-103`` -> (106.0, 103.0).
_CI_RE = re.compile(r"\+\s*([0-9.]+)\s*/\s*-\s*([0-9.]+)")

#: Metric key -> (website column, spec). Order = order in the y-axis selector;
#: the first entry is the default axis and the ranking used for the chips.
_METRIC_SPECS: dict[str, tuple[str, dict]] = {
    "elo": (
        "Elo [⬆️]",
        {
            "label": "Elo",
            "axisLabel": "Elo — higher is better",
            "lowerBetter": False,
            "fromZero": False,
            "decimals": 0,
            "suffix": "",
            "ci": {"lo": "elo_lo", "hi": "elo_hi"},
        },
    ),
    "imp": (
        "Improvability (%) [⬇️]",
        {
            "label": "Improvability (%)",
            "axisLabel": "Improvability (%) — lower is better",
            "lowerBetter": True,
            "fromZero": True,
            "decimals": 1,
            "suffix": "%",
        },
    ),
    "score": (
        "Score [⬆️]",
        {
            "label": "Score",
            "axisLabel": "Score — higher is better",
            "lowerBetter": False,
            "fromZero": True,
            "decimals": 3,
            "suffix": "",
        },
    ),
    "rank": (
        "Rank [⬇️]",
        {
            "label": "Average rank",
            "axisLabel": "Average rank — lower is better",
            "lowerBetter": True,
            "fromZero": True,
            "decimals": 2,
            "suffix": "",
        },
    ),
    "hrank": (
        "Harmonic Rank [⬇️]",
        {
            "label": "Harmonic rank",
            "axisLabel": "Harmonic rank — lower is better",
            "lowerBetter": True,
            "fromZero": True,
            "decimals": 2,
            "suffix": "",
        },
    ),
}


def _parse_model(model: str) -> tuple[str, str, str | None]:
    """Split a website ``Model`` cell into (method name, variant, url).

    The cell is ``[Name (variant) [X% IMPUTED]](url)`` with every part optional;
    a name whose parenthetical is not a tuning variant (e.g. the reference
    pipeline "AutoGluon 1.5 (extreme, 4h)") keeps it as part of the name.
    """
    link = re.match(r"\[(.*?)\]\((.*?)\)", model)
    text, url = (link.group(1), link.group(2)) if link else (model, None)
    text = text.split("[")[0].strip()  # drop any "[X% IMPUTED]" tag
    match = _VARIANT_RE.search(text)
    variant = _VARIANT_LABELS[match.group(1)] if match else ""
    return _VARIANT_RE.sub("", text).strip(), variant, url


def _parse_ci(value: object) -> tuple[float, float]:
    """``+106/-103`` -> the (upper, lower) offsets; (nan, nan) when absent."""
    match = _CI_RE.search(str(value)) if pd.notna(value) else None
    if not match:
        return float("nan"), float("nan")
    return float(match.group(1)), float(match.group(2))


def leaderboard_explorer_points(df_website_leaderboard: pd.DataFrame) -> pd.DataFrame:
    """The explorer's point frame: one row per method-variant of a website leaderboard."""
    df = df_website_leaderboard
    parsed = [_parse_model(m) for m in df["Model"]]
    points = pd.DataFrame(
        {
            "method": [p[0] for p in parsed],
            "variant": [p[1] for p in parsed],
            "family": df["TypeName"].to_numpy(),
            "url": [p[2] for p in parsed],
            "system": (df["TypeName"] == Constants.system).to_numpy(),
        },
    )
    for key, (column, _spec) in _METRIC_SPECS.items():
        points[key] = pd.to_numeric(df[column], errors="coerce").to_numpy() if column in df.columns else pd.NA

    if "Elo 95% CI" in df.columns:
        offsets = [_parse_ci(v) for v in df["Elo 95% CI"]]
        points["elo_hi"] = points["elo"] + [o[0] for o in offsets]
        points["elo_lo"] = points["elo"] - [o[1] for o in offsets]

    imputed_column = df["Imputed (%) [⬇️]"] if "Imputed (%) [⬇️]" in df.columns else pd.Series(0.0, index=df.index)
    imputed_pct = pd.to_numeric(imputed_column, errors="coerce").fillna(0.0)
    points["imputed_pct"] = imputed_pct.to_numpy()
    points["imputed"] = (imputed_pct > 0).to_numpy()
    return points.dropna(subset=["elo"]).reset_index(drop=True)


def build_leaderboard_explorer_html(
    df_website_leaderboard: pd.DataFrame,
    *,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena leaderboard explorer",
) -> Path | None:
    """Render the interactive leaderboard overview for one subset's website table.

    Parameters
    ----------
    df_website_leaderboard
        A ``website_leaderboard.csv`` frame (see
        :func:`tabarena.website.website_format.format_leaderboard`). Required
        columns: ``Model``, ``TypeName`` and ``Elo [⬆️]``; every other metric of
        :data:`_METRIC_SPECS` is offered on the y-axis selector when present.
    save_path
        Where to write the page.
    title
        Headline shown above the chart; omitted when ``None``.

    Returns:
    -------
    The written path, or ``None`` when the table has no usable Elo values.
    """
    for column in ("Model", "TypeName", "Elo [⬆️]"):
        if column not in df_website_leaderboard.columns:
            raise ValueError(f"website leaderboard is missing required column {column!r}")

    points = leaderboard_explorer_points(df_website_leaderboard)
    if points.empty:
        return None

    metrics = [
        {"key": key, **spec}
        for key, (column, spec) in _METRIC_SPECS.items()
        if column in df_website_leaderboard.columns and points[key].notna().any()
    ]
    # The CI only exists for Elo, and only when the table carries the column.
    for spec in metrics:
        if spec.get("ci") and "elo_hi" not in points.columns:
            spec.pop("ci")

    config = {"title": title, "metrics": metrics, "rankMetric": metrics[0]["key"]}
    html = render_explorer_html(LEADERBOARD_TEMPLATE, page_title=page_title, config=config, points=points)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
