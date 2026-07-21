"""Build the self-contained interactive Pareto explorer HTML.

The explorer is a single dependency-free HTML file (inline SVG + vanilla JS)
rendered from a points DataFrame. The leaderboard website embeds one such file
per subset in place of the static Pareto PNG; because the file is fully
self-contained it also works as a local artifact anyone can open in a browser.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.plot.interactive._explorer_template import EXPLORER_TEMPLATE

if TYPE_CHECKING:
    import pandas as pd

#: Variant display order used to sort each method's points so the connector
#: line runs default -> tuned -> tuned + ensembled.
_VARIANT_ORDER = ["Default", "Tuned", "Tuned + Ens.", "Baseline", "Best"]

#: Metric definitions the explorer's y-axis selector can offer, keyed by the
#: column name expected in ``points``.
_METRIC_SPECS: dict[str, dict] = {
    "imp": {
        "key": "imp",
        "label": "Improvability (%)",
        "axisLabel": "Improvability (%) — lower is better",
        "lowerBetter": True,
        "fromZero": True,
        "decimals": 1,
        "suffix": "%",
    },
    "elo": {
        "key": "elo",
        "label": "Elo",
        "axisLabel": "Elo — higher is better",
        "lowerBetter": False,
        "fromZero": False,
        "decimals": 0,
        "suffix": "",
    },
}

#: Time-axis definitions the explorer's x-axis selector can offer, keyed by
#: the column name expected in ``points``. All are log-scale seconds per 1K
#: samples (median over datasets).
_X_AXIS_SPECS: dict[str, dict] = {
    "x_infer": {
        "key": "x_infer",
        "label": "Inference time",
        "axisLabel": "Inference time per 1K samples (s), median — log scale",
        "short": "Inference (s/1K, median)",
    },
    "x_train": {
        "key": "x_train",
        "label": "Train time",
        "axisLabel": "Train time per 1K samples (s), median — log scale",
        "short": "Train (s/1K, median)",
    },
}


def build_pareto_explorer_html(
    points: pd.DataFrame,
    *,
    save_path: str | Path,
    mode: str = "scatter",
    title: str | None = None,
    metric_keys: list[str] | None = None,
    x_keys: list[str] | None = None,
    page_title: str = "TabArena Pareto explorer",
) -> Path:
    """Render the interactive explorer HTML for a set of method points.

    Parameters
    ----------
    points
        One row per plotted point. Required columns: ``method`` (display
        name), ``family`` (model family, e.g. "Foundation Model"), at least
        one x column (``x_infer`` and/or ``x_train``; positive, plotted on a
        log axis) and at least one metric column (``imp`` and/or ``elo``).
        Optional columns: ``variant`` (scatter mode:
        "Default"/"Tuned"/"Tuned + Ens."), ``n_configs`` (trajectory mode),
        ``imputed`` (bool) and ``imputed_pct`` (0-100).
    mode
        ``"scatter"`` (one point per method-variant, connectors link a
        method's variants) or ``"trajectory"`` (one line per method over
        ``n_configs``).
    metric_keys
        Which metrics of :data:`_METRIC_SPECS` to offer on the y-axis
        selector; defaults to every metric whose column is present.
    x_keys
        Which time axes of :data:`_X_AXIS_SPECS` to offer on the x-axis
        selector (first entry is the default view); defaults to every axis
        whose column is present.
    """
    if mode not in ("scatter", "trajectory"):
        raise ValueError(f"Unknown mode: {mode!r}")

    if metric_keys is None:
        metric_keys = [k for k in _METRIC_SPECS if k in points.columns]
    missing = [k for k in metric_keys if k not in points.columns]
    if not metric_keys or missing:
        raise ValueError(f"points must contain at least one metric column; missing={missing}, available={metric_keys}")
    if x_keys is None:
        x_keys = [k for k in _X_AXIS_SPECS if k in points.columns]
    missing_x = [k for k in x_keys if k not in points.columns]
    if not x_keys or missing_x:
        raise ValueError(f"points must contain at least one x column; missing={missing_x}, available={x_keys}")
    for col in ("method", "family"):
        if col not in points.columns:
            raise ValueError(f"points is missing required column {col!r}")

    data = points.copy()
    if "imputed" not in data.columns:
        data["imputed"] = False
    if "imputed_pct" not in data.columns:
        data["imputed_pct"] = 0.0
    data["imputed"] = data["imputed"].fillna(False).astype(bool)
    data["imputed_pct"] = data["imputed_pct"].fillna(0.0).astype(float)

    # The x-axes are logarithmic: rows without positive times cannot be placed.
    data = data.dropna(subset=[*x_keys, *metric_keys])
    for x_key in x_keys:
        data = data[data[x_key] > 0]

    # Point order per method defines the connector/trajectory line direction.
    if mode == "scatter":
        variant_rank = {v: i for i, v in enumerate(_VARIANT_ORDER)}
        if "variant" in data.columns:
            data["_rank"] = data["variant"].map(variant_rank).fillna(len(variant_rank))
        else:
            data["_rank"] = len(variant_rank)
    else:
        data["_rank"] = data["n_configs"] if "n_configs" in data.columns else data[x_keys[0]]
    data = data.sort_values(["method", "_rank"]).drop(columns=["_rank"]).reset_index(drop=True)

    keep_cols = ["method", "family", *x_keys, *metric_keys, "imputed", "imputed_pct"]
    for opt in ("variant", "n_configs"):
        if opt in data.columns:
            keep_cols.append(opt)
    data = data[keep_cols]

    config = {
        "mode": mode,
        "title": title,
        "metrics": [_METRIC_SPECS[k] for k in metric_keys],
        "xAxes": [_X_AXIS_SPECS[k] for k in x_keys],
    }

    html = (
        EXPLORER_TEMPLATE.replace("__PAGE_TITLE__", page_title)
        .replace("__CONFIG_JSON__", json.dumps(config))
        .replace("__POINTS_JSON__", data.to_json(orient="records"))
    )

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
