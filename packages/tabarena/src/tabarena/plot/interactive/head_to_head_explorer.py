"""Build the self-contained interactive head-to-head browser HTML.

The leaderboard says which of two methods is ahead on average; the per-dataset browser shows one
method against the whole field. This page sets two methods side by side, dataset by dataset: how
far apart their errors and ranks are on each, and whether the splits of that dataset can tell
them apart at all. The datasets can be sorted by any of those, so "where does A lose to B, and
is any of it more than noise" is one click.

Everything is computed in the page, for whichever two methods the reader picks, from the per-split
errors embedded here: the per-dataset significance (a two-sided Wilcoxon signed-rank test over the
dataset's paired splits, exact for up to 50 non-zero differences), and the leaderboard's Elo over
the datasets in view as a paired gap, A against the field without B and B against the field
without A on the same dataset resamples (:func:`elo_grid`), with each dataset's pull on that gap.
The page also has the gap for every subset of the datasets, A against every other method, and the
dataset's task metadata. Built from the per-split results the compare pass already wrote, like the
per-dataset browser, with every error rounded at its dataset's scale
(:func:`round_to_dataset_precision`) so that floating-point noise in a metric is never a win.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._head_to_head_template import HEAD_TO_HEAD_TEMPLATE
from tabarena.plot.interactive.per_dataset_explorer import (
    _METRIC_DISPLAY,
    _SIZE_BUCKETS,
    _drop_nulls,
    _round,
    _significant,
    dataset_records,
    measured_results,
    method_records,
)

#: Significance level of the per-dataset test, and of the page's "significant" counts.
SIGNIFICANCE_ALPHA = 0.05
#: Fewest non-zero paired differences the two-sided test is run on. Over ``n`` of them the
#: smallest two-sided p-value of the signed-rank test is ``2 / 2**n``, which is above 0.05 for
#: five or fewer: the test could never reject there, so those datasets report no verdict.
MIN_PAIRED_SPLITS = 6
#: Up to this many non-zero differences the page computes the exact null distribution of the
#: signed-rank statistic; above it, the normal approximation with the tie correction.
EXACT_MAX_PAIRS = 50
#: Dataset resamples behind the page's interval for the overall error ratio.
BOOTSTRAP_RESAMPLES = 2000
#: Dataset resamples behind the page's paired interval for the Elo gap between the two methods.
ELO_RESAMPLES = 1000
#: The method the page's Elo sets at 1000 unless the caller names the arena's own (TabArena's default;
#: BeyondArena pins ``XGB (default)``, see ``AbstractArenaContext.calibration_method``).
ELO_CALIBRATION = "RF (default)"
#: Significant digits an error keeps, counted at its dataset's typical error (see
#: :func:`round_to_dataset_precision`).
ERROR_DIGITS = 7


#: Column kinds a dataset can be filtered by, as BeyondArena's subsets of the same names draw them
#: (``BeyondArenaContext.SUBSET_PREDICATES``): ``(key, metadata columns, has the kind)``. A dataset
#: can be of several kinds; ``numerical`` is the one with no categorical, date or text column.
COLUMN_KINDS: list[tuple[str, tuple[str, ...], object]] = [
    ("text", ("num_text_cols",), lambda row: row["num_text_cols"] > 0),
    ("high-cardinality", ("num_high_cardinality_cats",), lambda row: row["num_high_cardinality_cats"] > 0),
    ("datetime", ("has_datetime",), lambda row: bool(row["has_datetime"])),
    ("categorical", ("has_categorical",), lambda row: bool(row["has_categorical"])),
    (
        "numerical",
        ("has_categorical", "has_datetime", "num_text_cols"),
        lambda row: not bool(row["has_categorical"]) and not bool(row["has_datetime"]) and row["num_text_cols"] == 0,
    ),
]


#: The task metadata the selected dataset's pane lists, each where the metadata has it:
#: ``(column, label, format, flag)``. ``flag`` is a yes / no column shown in the count's place when
#: the count is absent: TabArena's stored metadata has the flags, and a caller can merge in the
#: counts (``num_numerical_cols`` and the like, by the same column rules) for a finer view.
DATASET_FACTS: list[tuple[str, str, str, str | None]] = [
    ("num_instances", "Rows", "count", None),
    ("num_features", "Features", "count", None),
    ("num_numerical_cols", "Numerical columns", "count", "has_numerical"),
    ("num_categorical_cols", "Categorical columns", "count", "has_categorical"),
    ("num_high_cardinality_cats", "Categoricals with > 50 levels", "count", "has_high_cardinality_categorical"),
    ("max_categorical_levels", "Most levels of a categorical", "count", None),
    ("num_binary_cols", "Binary columns", "count", "has_binary"),
    ("num_datetime_cols", "Date columns", "count", "has_datetime"),
    ("num_text_cols", "Text columns", "count", "has_text"),
    ("num_cols_after_preprocessing", "Columns after preprocessing", "count", None),
    ("missing_value_fraction", "Missing cells", "percent", None),
    ("num_cols_with_missing", "Columns with missing values", "count", None),
    ("num_classes", "Classes", "count", None),
    ("target_imbalance_ratio", "Class ratio (majority : minority)", "ratio", None),
    ("target_skewness", "Target skewness", "number", None),
    ("num_instance_groups", "Groups", "count", None),
    ("group_on", "Group column", "text", None),
    ("time_on", "Time column", "text", None),
    ("domain", "Domain", "text", None),
    ("source", "Source", "text", None),
    ("dataset_year", "Year", "text", None),
]


#: How each format of :data:`DATASET_FACTS` writes a value.
_FACT_FORMATS = {
    "count": lambda value: f"{int(value):,}",
    "percent": lambda value: f"{float(value) * 100:.2f}%",
    "ratio": lambda value: f"{float(value):.1f} : 1",
    "number": lambda value: f"{float(value):.2f}",
    "text": str,
}


def _fact_value(value: object, fmt: str) -> str | None:
    """A metadata value as the pane shows it, or ``None`` when it is missing."""
    if isinstance(value, (list, tuple)):
        return ", ".join(str(v) for v in value) or None
    if value is None or pd.isna(value):
        return None
    return _FACT_FORMATS[fmt](value)


def _uninformative(column: str, row: dict) -> bool:
    """A value the metadata stores that says nothing: the class count of a regression task (-1), or a group
    count equal to the row count (a task without groups has one per row).
    """
    value = row.get(column)
    if column == "num_classes":
        return value is not None and not pd.isna(value) and value < 2
    if column == "num_instance_groups":
        return value is not None and not pd.isna(value) and value == row.get("num_instances")
    return False


def dataset_facts(dataset_metadata: pd.DataFrame | None) -> dict[str, list[list[str]]]:
    """Each dataset's ``[label, value]`` pairs of :data:`DATASET_FACTS`, for the columns its metadata has."""
    if dataset_metadata is None or "dataset" not in dataset_metadata.columns:
        return {}
    columns = set(dataset_metadata.columns)
    out = {}
    for row in dataset_metadata.to_dict("records"):
        pairs = []
        for column, label, fmt, flag in DATASET_FACTS:
            value = _fact_value(row[column], fmt) if column in columns and not _uninformative(column, row) else None
            if value is None and flag in columns and not pd.isna(row[flag]):
                value = "yes" if bool(row[flag]) else "no"
            if value is not None:
                pairs.append([label, value])
        out[row["dataset"]] = pairs
    return out


#: Task metadata the chart can put on its x-axis, each where the metadata has it:
#: ``(column, key, label, scale)``. ``"log1p"`` is a log scale that keeps 0 (counts that can be 0),
#: ``"log"`` one for values above 0, ``"percent"`` a linear scale for a fraction.
DATASET_AXES: list[tuple[str, str, str, str]] = [
    ("num_categorical_cols", "cat", "Categorical columns", "log1p"),
    ("max_categorical_levels", "lev", "Most levels of a categorical", "log1p"),
    ("num_high_cardinality_cats", "hc", "Categoricals with > 50 levels", "log1p"),
    ("num_text_cols", "txt", "Text columns", "log1p"),
    ("num_datetime_cols", "dt", "Date columns", "log1p"),
    ("num_binary_cols", "bin", "Binary columns", "log1p"),
    ("num_numerical_cols", "numc", "Numerical columns", "log1p"),
    ("num_cols_after_preprocessing", "cols", "Columns after preprocessing (est.)", "log"),
    ("missing_value_fraction", "miss", "Missing cells", "percent"),
    ("target_imbalance_ratio", "imb", "Class ratio (majority : minority)", "log"),
]


def dataset_axes(dataset_metadata: pd.DataFrame | None) -> tuple[list[dict], dict[str, dict[str, float]]]:
    """The :data:`DATASET_AXES` the metadata has a value for on some dataset, and each dataset's values by key."""
    if dataset_metadata is None or "dataset" not in dataset_metadata.columns:
        return [], {}
    present = [axis for axis in DATASET_AXES if axis[0] in dataset_metadata.columns]
    values: dict[str, dict[str, float]] = {}
    for row in dataset_metadata.to_dict("records"):
        found = {key: float(row[column]) for column, key, _, _ in present if not pd.isna(row[column])}
        if found:
            values[row["dataset"]] = found
    used = {key for found in values.values() for key in found}
    axes = [{"key": key, "label": label, "scale": scale} for _, key, label, scale in present if key in used]
    return axes, values


def column_kinds(dataset_metadata: pd.DataFrame | None) -> dict[str, list[str]]:
    """Each dataset's column kinds (keys of :data:`COLUMN_KINDS`), for the kinds its metadata can tell.

    Empty when the metadata has none of the columns, as for an arena without column statistics.
    """
    if dataset_metadata is None or "dataset" not in dataset_metadata.columns:
        return {}
    usable = [kind for kind in COLUMN_KINDS if all(column in dataset_metadata.columns for column in kind[1])]
    if not usable:
        return {}
    # A missing statistic (None / NaN) tells nothing about its kinds, so they are left out for that dataset.
    return {
        row["dataset"]: [
            key for key, needed, has in usable if not any(pd.isna(row[column]) for column in needed) and has(row)
        ]
        for row in dataset_metadata.to_dict("records")
    }


def round_to_dataset_precision(results_per_split: pd.DataFrame, digits: int = ERROR_DIGITS) -> pd.DataFrame:
    """``results_per_split`` with every error rounded to the same decimal place across its dataset.

    The place is the ``digits``-th significant digit of the dataset's typical error, the median
    non-zero error over its methods and splits. Two methods whose predictions score the same can
    still differ in the metric's last bits: by the order of a sum, or as a ``1 - AUC`` of 1.1e-16
    (an AUC one bit short of 1) where another method gets exactly 0. Rounding each value to its
    own significant digits keeps that difference whenever one side is 0, and the page would count
    a won split, a better rank and a lower mean out of it; rounding at the dataset's scale makes
    it the tie it is. A dataset whose errors are all 0 is left as it is.
    """
    df = results_per_split.copy()
    errors = df["metric_error"].abs()
    scale = errors.where(errors > 0).groupby(df["dataset"]).transform("median")
    decimals = digits - 1 - np.floor(np.log10(scale))
    for places, index in df.groupby(decimals).groups.items():
        df.loc[index, "metric_error"] = df.loc[index, "metric_error"].round(int(places))
    return df


def head_to_head_points(results_per_split: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """One row per (dataset, method) with its summary and its error on every split, plus each dataset's splits.

    ``err``, ``std``, ``rank``, ``imp`` and ``train_s`` are as in the per-dataset browser (see
    :func:`tabarena.plot.interactive.per_dataset_explorer.per_dataset_points`): the rank is among
    the whole field of the cell, not between the two methods. ``splits`` holds the method's
    error on each of the dataset's splits, in the order of the returned ``{dataset: [fold, ...]}``,
    which is what the page pairs two methods on. Only methods measured on every split of a dataset
    are kept for it (:func:`~tabarena.plot.interactive.per_dataset_explorer.measured_results`).
    Every error is first rounded at its dataset's scale (:func:`round_to_dataset_precision`), so
    the means, ranks and split errors all treat floating-point noise as a tie.
    """
    return _points(round_to_dataset_precision(results_per_split))


def _points(rounded: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, list[int]]]:
    """:func:`head_to_head_points` of results already rounded at each dataset's scale."""
    df = measured_results(rounded)
    best_per_split = df.groupby(["dataset", "fold"])["metric_error"].transform("min")
    df["_imp"] = (1 - best_per_split / df["metric_error"]).fillna(0.0) * 100
    df["_rank"] = df.groupby(["dataset", "fold"])["metric_error"].rank(method="average")
    points = df.groupby(["dataset", "method"], as_index=False).agg(
        err=("metric_error", "mean"),
        std=("metric_error", "std"),
        rank=("_rank", "mean"),
        imp=("_imp", "mean"),
        train_s=("time_train_s", "mean"),
        infer_s=("time_infer_s", "mean"),
    )
    folds = {dataset: sorted(int(f) for f in group.unique()) for dataset, group in df.groupby("dataset")["fold"]}
    # Every kept method ran on every split of its dataset, so its errors in fold order line up
    # with the dataset's fold list.
    per_split = (
        df.groupby(["dataset", "method", "fold"], as_index=False)["metric_error"]
        .mean()
        .sort_values(["dataset", "method", "fold"])
        .groupby(["dataset", "method"])["metric_error"]
        .agg(lambda errors: [None if pd.isna(v) else float(v) for v in errors])
    )
    points["splits"] = [
        per_split[(dataset, method)] for dataset, method in zip(points["dataset"], points["method"], strict=True)
    ]
    return points, folds


def elo_grid(
    rounded: pd.DataFrame,
    folds: dict[str, list[int]],
    method_index: dict[str, int],
) -> tuple[list[int], list[list[list[float]]]]:
    """The field the page fits its Elo on, and every member's error on every split of every dataset.

    The field is every method with a result on every split, imputed results included, as on the
    leaderboard (bencheval's rank path needs that complete grid). Returns the field's method
    indices and, per dataset in ``folds`` order, one row of split errors per field method.
    """
    units = pd.MultiIndex.from_tuples(
        [(dataset, fold) for dataset, dataset_folds in folds.items() for fold in dataset_folds],
        names=["dataset", "fold"],
    )
    wide = rounded.pivot_table(index="method", columns=["dataset", "fold"], values="metric_error", aggfunc="mean")
    wide = wide.reindex(columns=units)
    field = [method for method in wide.index[wide.notna().all(axis=1)] if method in method_index]
    grid = [wide.loc[field, dataset].to_numpy().tolist() for dataset in folds]
    return [method_index[method] for method in field], grid


def build_head_to_head_explorer_html(
    *,
    results_per_split: pd.DataFrame,
    method_info: pd.DataFrame,
    dataset_metadata: pd.DataFrame | None = None,
    method_a: str | None = None,
    method_b: str | None = None,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena head-to-head",
    size_buckets: list[tuple[str, str, float]] | None = None,
    elo_calibration: str | None = ELO_CALIBRATION,
) -> Path | None:
    """Render the head-to-head browser for one subset.

    Parameters
    ----------
    results_per_split
        The cell's ``results_per_split.csv``: one row per (dataset, split, method) with
        ``metric_error`` and the runtimes.
    method_info
        The cell's ``method_info.csv``, which names and classifies every method in it.
    dataset_metadata
        One row per dataset (``TaskMetadataCollection.per_dataset_frame``), used for the size,
        task and split filters and the dataset facts.
    method_a, method_b
        Display names of the two methods the page opens on (the leaderboard's names, e.g.
        ``"CatBoost (tuned + ensemble)"``). Each falls back to the best mean rank among the
        methods not already picked, so by default the page opens on the field's top two.
    title
        Headline shown above the controls; omitted when ``None``.
    size_buckets
        ``(key, label, max_train_rows)`` per size chip, matching the arena's own size subsets;
        defaults to TabArena's small / medium / large.
    elo_calibration
        The method (as the results name it) the page's Elo sets at 1000, as the leaderboard does;
        without it in the field the ratings average 1000. The page's Elo gap between A and B is
        paired: A against the field without B and B against the field without A, both on the same
        dataset resamples (see :func:`elo_grid`).

    Returns:
    -------
    The written path, or ``None`` when the results frame holds fewer than two usable methods.
    """
    methods = method_records(method_info)
    datasets = dataset_records(results_per_split, dataset_metadata)
    rounded = round_to_dataset_precision(results_per_split)
    points, folds = _points(rounded)
    points = points[points["method"].isin(set(methods["method"]))]
    # Every method of the results, also one without a fully measured dataset: it has no rows to
    # compare, but it is part of the field the Elo is fitted on.
    methods = methods[methods["method"].isin(set(rounded["method"]))].reset_index(drop=True)
    datasets = datasets[datasets["dataset"].isin(folds)].reset_index(drop=True)
    if points["method"].nunique() < 2 or datasets.empty:
        return None
    folds = {dataset: folds[dataset] for dataset in datasets["dataset"]}

    dataset_index = {key: i for i, key in enumerate(datasets["dataset"])}
    method_index = {key: i for i, key in enumerate(methods["method"])}
    records = pd.DataFrame(
        {
            "d": points["dataset"].map(dataset_index).to_numpy(),
            "m": points["method"].map(method_index).to_numpy(),
            "e": [_significant(v) for v in points["err"]],
            "s": [_significant(v, 3) for v in points["std"]],
            "r": [_round(v, 2) for v in points["rank"]],
            "i": [_round(v, 3) for v in points["imp"]],
            "t": [_significant(v, 4) for v in points["train_s"]],
            "it": [_significant(v, 4) for v in points["infer_s"]],
        },
    )
    elo_field, grid = elo_grid(rounded, folds, method_index)
    # A field member's split errors are its rows of the Elo grid; only the others carry their own.
    in_field = set(elo_field)
    records["f"] = [None if m in in_field else splits for m, splits in zip(records["m"], points["splits"], strict=True)]

    # Fallback order for an unnamed or unknown pick: the mean rank over the datasets each method
    # ran on, which is the per-dataset browser's fallback too.
    by_rank = points.groupby("method")["rank"].mean().sort_values().index.map(method_index).tolist()
    by_name = {name: i for i, name in enumerate(methods["name"])}
    picks: list[int] = []
    for wanted in (method_a, method_b):
        index = by_name.get(wanted) if wanted else None
        if index is None or index in picks:
            index = next(i for i in by_rank if i not in picks)
        picks.append(index)

    dataset_list = _drop_nulls(datasets.drop(columns=["dataset"]).assign(key=datasets["dataset"]))
    kinds = column_kinds(dataset_metadata)
    facts = dataset_facts(dataset_metadata)
    x_axes, x_values = dataset_axes(dataset_metadata)
    for record, key in zip(dataset_list, datasets["dataset"], strict=True):
        record["folds"] = folds[key]
        if key in kinds:
            record["kinds"] = kinds[key]
        if facts.get(key):
            record["facts"] = facts[key]
        if key in x_values:
            record["x"] = x_values[key]
    config = {
        "title": title,
        "datasets": dataset_list,
        "methods": _drop_nulls(methods.drop(columns=["method"])),
        "sizeBuckets": [
            {"key": key, "label": label, "max": None if max_rows == math.inf else max_rows}
            for key, label, max_rows in (_SIZE_BUCKETS if size_buckets is None else size_buckets)
        ],
        "metricDisplay": _METRIC_DISPLAY,
        "xAxes": x_axes,
        "defaultA": int(picks[0]),
        "defaultB": int(picks[1]),
        "significance": {
            "alpha": SIGNIFICANCE_ALPHA,
            "minSplits": MIN_PAIRED_SPLITS,
            "exactMax": EXACT_MAX_PAIRS,
            "test": "two-sided Wilcoxon signed-rank test over the paired splits",
        },
        "bootstrap": BOOTSTRAP_RESAMPLES,
        "elo": {
            "field": elo_field,
            "grid": grid,
            "calibration": method_index.get(elo_calibration) if elo_calibration else None,
            "resamples": ELO_RESAMPLES,
        }
        if len(elo_field) >= 2
        else None,
    }
    html = render_explorer_html(HEAD_TO_HEAD_TEMPLATE, page_title=page_title, config=config, points=records)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path
