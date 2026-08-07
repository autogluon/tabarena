"""Build the self-contained interactive per-dataset browser HTML.

The leaderboard answers "which method is best on average". This answers the question a reader
asks next: *where* did it win, and where did it lose. One row per dataset, a contender the
reader picks, and — for the dataset they select — the tuning trajectories of the whole field on
that dataset alone.

Like the leaderboard table and overview it is built in the artifact-conversion step, from the
per-split results the evaluation already wrote plus the per-dataset trajectory frame from
:func:`tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time.compute_per_dataset_trajectories`.
Nothing here recomputes a leaderboard, so a styling change costs a conversion re-run rather
than a re-evaluation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pandas as pd

from tabarena.models._method_metadata import MethodMetadata
from tabarena.plot.interactive._explorer_shared import render_explorer_html
from tabarena.plot.interactive._per_dataset_template import PER_DATASET_TEMPLATE
from tabarena.website.website_format import Constants, get_model_family

#: ``method_subtype`` -> the variant label the rest of the site uses.
_VARIANT_LABELS = {"default": "Default", "tuned": "Tuned", "tuned_ensemble": "Tuned + Ens."}

#: Eval-metric key -> the short name shown on the error axis. ``metric_error`` for ROC AUC is
#: ``1 - AUC``, which is worth spelling out. Mirrors the static figures' y-axis labels.
_METRIC_DISPLAY = {
    "log_loss": "logloss",
    "roc_auc": "1-AUC",
    "rmse": "RMSE",
    "root_mean_squared_error": "RMSE",
}

#: Training-row buckets, keyed on ``max_train_rows`` exactly as the site's dataset-size subsets
#: are (see ``TabArenaContext.SUBSET_PREDICATES``), so the browser's own size filter and the
#: leaderboard's "Small" / "Medium" tabs cannot disagree about where a dataset belongs.
_SIZE_BUCKETS: list[tuple[str, str, float]] = [
    ("small", "≤ 10k rows", 10_000),
    ("medium", "10k – 100k", 100_000),
    ("large", "> 100k", math.inf),
]

#: Columns of ``CONFIG.trajectory.rows``, kept as positional arrays rather than records: the
#: frame is one row per (dataset, method, tuning budget) and the key names would outweigh the
#: numbers several times over. ``x`` is the median training time in seconds on that dataset;
#: the per-1K normalization the aggregate figure needs only rescales a single dataset's axis.
_TRAJECTORY_COLS = ["d", "m", "n", "x", "e", "i"]


def _round(value: object, digits: int) -> float | None:
    """Round for JSON, mapping anything non-finite to ``None`` (which reads as ``null``)."""
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number):
        return None
    return round(number, digits)


def _significant(value: object, digits: int = 6) -> float | None:
    """Round to ``digits`` significant figures, so a 1e-4 error keeps its precision and a
    four-digit runtime does not carry ten decimals into the page.
    """
    if value is None or pd.isna(value):
        return None
    number = float(value)
    if not math.isfinite(number) or number == 0:
        return 0.0 if number == 0 else None
    exponent = math.floor(math.log10(abs(number)))
    return round(number, max(0, digits - 1 - exponent))


def method_records(method_info: pd.DataFrame) -> pd.DataFrame:
    """One row per benchmarked method, with the identity fields the browser renders.

    ``base`` is the method's name without its tuning variant, which is what the per-dataset
    trajectory frame is keyed on: picking "CatBoost (tuned)" as the contender has to light up
    the CatBoost trajectory.
    """
    info = method_info.copy()
    if "method_class" not in info.columns:
        info["method_class"] = "model"
    systems = frozenset(info.loc[info["method_class"] == "system", "display_name"].dropna())

    names, bases, variants, families, urls = [], [], [], [], []
    for row in info.to_dict("records"):
        config_type = row.get("config_type")
        display_name = row.get("display_name")
        names.append(
            MethodMetadata.compute_method_name(
                method=row["method"],
                method_type=row["method_type"],
                method_subtype=row.get("method_subtype") if pd.notna(row.get("method_subtype")) else None,
                config_type=config_type if pd.notna(config_type) else None,
                display_name=display_name if pd.notna(display_name) else None,
            ),
        )
        base = display_name if pd.notna(display_name) else (config_type if pd.notna(config_type) else row["method"])
        bases.append(base)
        subtype = row.get("method_subtype")
        variants.append(_VARIANT_LABELS.get(subtype, "") if pd.notna(subtype) else "")
        if row.get("method_class") == "system":
            families.append(Constants.system)
        else:
            families.append(get_model_family(base, system_names=systems))
        url = row.get("reference_url")
        urls.append(url if pd.notna(url) else None)

    return pd.DataFrame(
        {
            "method": info["method"].to_numpy(),
            "name": names,
            "base": bases,
            "variant": variants,
            "family": families,
            "url": urls,
        },
    )


def dataset_records(
    results_per_split: pd.DataFrame,
    dataset_metadata: pd.DataFrame | None,
) -> pd.DataFrame:
    """One row per dataset in the results, with whatever task metadata is available.

    The results frame is the authority on *which* datasets exist (it is the cell being
    published); ``dataset_metadata`` only enriches them. A missing metadata frame therefore
    degrades to a browser with fewer columns rather than an empty one.
    """
    from_results = (
        results_per_split.groupby("dataset", as_index=False)
        .agg(task=("problem_type", "first"), metric=("metric", "first"), splits=("fold", "nunique"))
        .sort_values("dataset", ignore_index=True)
    )
    if dataset_metadata is None or "dataset" not in dataset_metadata.columns:
        from_results["name"] = from_results["dataset"]
        return from_results

    columns = {
        "dataset_name": "name",
        "num_instances": "rows",
        "num_features": "features",
        "num_classes": "classes",
        "max_train_rows": "train_rows",
        "domain": "domain",
        "source": "source",
        "dataset_year": "year",
    }
    available = {src: dst for src, dst in columns.items() if src in dataset_metadata.columns}
    meta = dataset_metadata[["dataset", *available]].rename(columns=available)
    merged = from_results.merge(meta, on="dataset", how="left")
    if "name" in merged.columns:
        merged["name"] = merged["name"].fillna(merged["dataset"])
    else:
        merged["name"] = merged["dataset"]
    return merged


def per_dataset_points(results_per_split: pd.DataFrame) -> pd.DataFrame:
    """One row per (dataset, method): how that method did on that dataset alone.

    ``rank`` and ``imp`` are computed per split and then averaged, matching how the leaderboard
    aggregates them (:meth:`bencheval.evaluator.BenchmarkEvaluator.compute_improvability_per`),
    so a dataset's numbers here are the per-dataset terms of the leaderboard's averages.
    """
    df = results_per_split.copy()
    best_per_split = df.groupby(["dataset", "fold"])["metric_error"].transform("min")
    df["_imp"] = (1 - best_per_split / df["metric_error"]).fillna(0.0) * 100
    df["_rank"] = df.groupby(["dataset", "fold"])["metric_error"].rank(method="average")
    return df.groupby(["dataset", "method"], as_index=False).agg(
        err=("metric_error", "mean"),
        rank=("_rank", "mean"),
        imp=("_imp", "mean"),
        train_s=("time_train_s", "mean"),
        imputed=("imputed", "max"),
    )


def build_per_dataset_explorer_html(
    *,
    results_per_split: pd.DataFrame,
    method_info: pd.DataFrame,
    trajectories: pd.DataFrame | None = None,
    dataset_metadata: pd.DataFrame | None = None,
    save_path: str | Path,
    title: str | None = None,
    page_title: str = "TabArena per-dataset results",
) -> Path | None:
    """Render the per-dataset browser for one subset.

    Parameters
    ----------
    results_per_split
        The cell's ``results_per_split.csv``: one row per (dataset, split, method) with
        ``metric_error`` and the runtimes.
    method_info
        The cell's ``method_info.csv``, which names and classifies every method in it.
    trajectories
        ``tuning_trajectories_per_dataset.csv`` (see
        :func:`~tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time.compute_per_dataset_trajectories`).
        Without it the browser still renders; selecting a dataset then shows its ranking
        without the trajectory chart.
    dataset_metadata
        One row per dataset (``TaskMetadataCollection.per_dataset_frame``), used for the size
        and task filters and the selected dataset's metadata line.
    title
        Headline shown above the table; omitted when ``None``.

    Returns:
    -------
    The written path, or ``None`` when the results frame holds no usable methods.
    """
    methods = method_records(method_info)
    datasets = dataset_records(results_per_split, dataset_metadata)
    points = per_dataset_points(results_per_split)
    points = points[points["method"].isin(set(methods["method"]))]
    if points.empty or datasets.empty:
        return None

    # Only methods and datasets that actually appear in this cell get an index, so the page
    # never offers a contender with no rows.
    methods = methods[methods["method"].isin(set(points["method"]))].reset_index(drop=True)
    dataset_index = {key: i for i, key in enumerate(datasets["dataset"])}
    method_index = {key: i for i, key in enumerate(methods["method"])}

    records = pd.DataFrame(
        {
            "d": points["dataset"].map(dataset_index).to_numpy(),
            "m": points["method"].map(method_index).to_numpy(),
            "e": [_significant(v) for v in points["err"]],
            "r": [_round(v, 2) for v in points["rank"]],
            "i": [_round(v, 3) for v in points["imp"]],
            "t": [_significant(v, 4) for v in points["train_s"]],
            "q": points["imputed"].fillna(0).astype(bool).to_numpy(),
        },
    )

    trajectory_methods: list[str] = []
    trajectory_rows: list[list[float | int | None]] = []
    if trajectories is not None and not trajectories.empty:
        usable = trajectories[trajectories["dataset"].isin(dataset_index)]
        trajectory_methods = sorted(usable["method"].dropna().unique().tolist())
        traj_index = {name: i for i, name in enumerate(trajectory_methods)}
        trajectory_rows = [
            [
                dataset_index[row["dataset"]],
                traj_index[row["method"]],
                None if pd.isna(row["n_configs"]) else int(row["n_configs"]),
                _significant(row["train_s"], 4),
                _significant(row["err"]),
                _round(row["imp"], 3),
            ]
            for row in usable.to_dict("records")
        ]

    # The contender the page opens on: the method with the best mean rank across this cell's
    # datasets. Whoever leads the leaderboard is the one a reader is most likely to be checking
    # for weak spots, and it makes the first paint informative without a click.
    mean_rank = points.groupby("method")["rank"].mean()
    default_contender = method_index[mean_rank.idxmin()] if not mean_rank.empty else 0

    config = {
        "title": title,
        "datasets": _drop_nulls(datasets.drop(columns=["dataset"]).assign(key=datasets["dataset"])),
        "methods": _drop_nulls(methods.drop(columns=["method"])),
        "trajectoryMethods": trajectory_methods,
        "trajectory": {"cols": _TRAJECTORY_COLS, "rows": trajectory_rows},
        "sizeBuckets": [
            {"key": key, "label": label, "max": None if max_rows == math.inf else max_rows}
            for key, label, max_rows in _SIZE_BUCKETS
        ],
        "metricDisplay": _METRIC_DISPLAY,
        "defaultContender": int(default_contender),
    }
    html = render_explorer_html(PER_DATASET_TEMPLATE, page_title=page_title, config=config, points=records)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(html, encoding="utf-8")
    return save_path


def _drop_nulls(df: pd.DataFrame) -> list[dict]:
    """Records with the missing fields left out entirely, rather than carried as nulls.

    Every absent value would otherwise cost its key name in the page for all 51 datasets.
    Values are unwrapped to Python scalars on the way out, since ``json.dumps`` cannot
    serialize the numpy types a numeric column yields.
    """
    return [
        {key: (value.item() if hasattr(value, "item") else value) for key, value in row.items() if pd.notna(value)}
        for row in df.to_dict("records")
    ]
