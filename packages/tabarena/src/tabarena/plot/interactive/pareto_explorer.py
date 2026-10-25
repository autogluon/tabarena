"""Build the self-contained interactive Pareto explorer HTML.

The explorer is a single dependency-free HTML file (inline SVG + vanilla JS)
rendered from a points DataFrame. The leaderboard website embeds one such file
per subset in place of the static Pareto PNG; because the file is fully
self-contained it also works as a local artifact anyone can open in a browser.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._explorer_template import EXPLORER_TEMPLATE

if TYPE_CHECKING:
    import pandas as pd

#: Variant display order used to sort each method's points so the connector
#: line runs default -> tuned -> tuned + ensembled.
_VARIANT_ORDER = ["Default", "Tuned", "Tuned + Ens.", "End-to-end", "Best"]

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

#: The explorer's time measures: column stem, selector label, axis-label stem, short (tooltip / table) stem.
_TIME_KINDS = {
    "total": ("Total time", "Total (train + inference) time", "Total"),
    "infer": ("Inference time", "Inference time", "Inference"),
    "train": ("Train time", "Train time", "Train"),
}


def _x_axis_spec(kind: str, per_task: bool, mean: bool) -> dict:
    """The spec of one time axis: ``kind`` of `_TIME_KINDS`, per task or per 1K samples, mean or median over datasets.

    Keys are ``x_<kind>`` (median per 1K samples) plus ``_task`` (per task) and ``_mean`` (mean over datasets).
    """
    label, axis_stem, short_stem = _TIME_KINDS[kind]
    stat = "mean" if mean else "median"
    unit, unit_short = ("per task", "s/task") if per_task else ("per 1K samples", "s/1K")
    return {
        "key": f"x_{kind}" + ("_task" if per_task else "") + ("_mean" if mean else ""),
        "label": f"{label} ({unit}, {stat})",  # the closed selector shows the option alone, not its group
        "group": f"{stat.capitalize()}, {unit}",
        "axisLabel": f"{axis_stem} {unit} (s), {stat} over datasets — log scale",
        "short": f"{short_stem} ({unit_short}, {stat})",
    }


#: Time-axis definitions the explorer's x-axis selector can offer, keyed by the column name expected in ``points``: the
#: inference, train and total (train + inference) time of a task, per 1K samples or per task, its median or mean over
#: datasets (a dataset's splits averaged first), all on a log scale. The first, ``x_total`` (total time per 1K
#: samples, median), is what a page shows at load when every axis is offered.
_X_AXIS_SPECS: dict[str, dict] = {
    spec["key"]: spec
    for per_task in (False, True)
    for mean in (False, True)
    for kind in _TIME_KINDS
    for spec in [_x_axis_spec(kind, per_task, mean)]
}
#: The time-axis keys in the selector's order.
X_AXIS_KEYS: tuple[str, ...] = tuple(_X_AXIS_SPECS)


def build_pareto_explorer_html(
    points: pd.DataFrame,
    *,
    save_path: str | Path,
    mode: str = "scatter",
    title: str | None = None,
    metric_keys: list[str] | None = None,
    x_keys: list[str] | None = None,
    subsets: list[tuple[str, str]] | None = None,
    chips_side: str = "left",
    dim_off_front: bool = False,
    page_title: str = "TabArena Pareto explorer",
) -> Path:
    """Render the interactive explorer HTML for a set of method points.

    Parameters
    ----------
    points
        One row per plotted point. Required columns: ``method`` (display
        name), ``family`` (model family, e.g. "Foundation Model"), at least
        one x column (a key of :data:`_X_AXIS_SPECS`, e.g. ``x_infer`` or
        ``x_train``; positive, plotted on a log axis) and at least one metric
        column (``imp`` and/or ``elo``).
        Optional columns: ``variant`` (scatter mode:
        "Default"/"Tuned"/"Tuned + Ens."), ``n_configs`` (trajectory mode),
        ``imputed`` (bool), ``imputed_pct`` (0-100) and ``shade`` (a system's
        shade, see :func:`tabarena.website.website_format.system_shade`).
    mode
        ``"scatter"`` (one point per method-variant, connectors link a
        method's variants) or ``"trajectory"`` (one line per method over
        ``n_configs``).
    title
        The figure title drawn above the chart, joined with the shown
        subset's label when the page holds ``subsets``; the page's title box
        edits it per subset, and an emptied box draws none. ``None`` (the
        default) leaves only the subset label, or no title.
    metric_keys
        Which metrics of :data:`_METRIC_SPECS` to offer on the y-axis
        selector; defaults to every metric whose column is present.
    x_keys
        Which time axes of :data:`_X_AXIS_SPECS` the page's x-axis selector
        offers, the first shown at load (one axis hides the selector);
        defaults to every axis whose column is present.
    subsets
        ``(key, label)`` pairs of the task subsets ``points`` holds (its
        ``subset`` column names each point's key): the page's subset
        selector shows one at a time, the first at load. ``None`` (the
        default): ``points`` is one subset and the page has no selector.
    chips_side
        ``"left"`` places the controls + method chips left of the chart,
        ``"right"`` mirrors the columns.
    dim_off_front
        Draw the active points that are not vertices of the current Pareto front
        smaller and fainter, so the front members stand out among a large selection.
    """
    if chips_side not in ("left", "right"):
        raise ValueError(f"Unknown chips_side: {chips_side!r}")
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
    for col in ("method", "family", *(["subset"] if subsets else [])):
        if col not in points.columns:
            raise ValueError(f"points is missing required column {col!r}")
    if subsets:
        unknown = set(points["subset"]) - {key for key, _ in subsets}
        if unknown:
            raise ValueError(f"points name subsets the `subsets` list lacks: {sorted(unknown)}")

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
    for opt in ("variant", "n_configs", "shade", *(["subset"] if subsets else [])):
        if opt in data.columns:
            keep_cols.append(opt)
    data = data[keep_cols]

    config = {
        "mode": mode,
        "title": title,
        "metrics": [_METRIC_SPECS[k] for k in metric_keys],
        "xAxes": [_X_AXIS_SPECS[k] for k in x_keys],
        "subsets": [{"key": key, "label": label} for key, label in subsets] if subsets else None,
        "chipsSide": chips_side,
        "dimOffFront": dim_off_front,
    }

    html = render_explorer_html(EXPLORER_TEMPLATE, page_title=page_title, config=config, points=data)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
