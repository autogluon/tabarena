from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
from tabarena.contexts.tabarena_context import TabArenaContext
from tabarena.nips2025_utils.subset_predicate import SubsetPredicate
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator

if TYPE_CHECKING:
    from collections.abc import Callable


def compare(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    task_metadata: TaskMetadataCollection,
    only_valid_tasks: str | list[str] | None = None,
    calibration_framework: str | None = None,
    fillna: str | pd.DataFrame | None = None,
    score_on_val: bool = False,
    average_seeds: bool = False,
    leaderboard_kwargs: dict | None = None,
    remove_imputed: bool = False,
    method_rename_map: dict | None = None,
    figure_file_type: str = "pdf",
    add_dataset_count: bool = False,
    elo_ymin: float | None = None,
    benchmark_name: str = "Arena",
    **kwargs,
):
    """Evaluate ``df_results`` (already subset to the tasks of interest) into a leaderboard.

    Generic over the supplied ``task_metadata``; task subsetting is the caller's job
    (see :meth:`AbstractArenaContext.compare`, which subsets via ``subset_results`` first).
    """
    df_results = prepare_data(
        df_results=df_results,
        only_valid_tasks=only_valid_tasks,
        fillna=fillna,
        remove_imputed=remove_imputed,
    )

    if score_on_val:
        error_col = "metric_error_val"
        df_results = df_results[~df_results["metric_error_val"].isna()]
    else:
        error_col = "metric_error"

    if calibration_framework == "auto":
        calibration_framework = None if isinstance(fillna, pd.DataFrame) else fillna

    evaluator_kwargs = {}
    if elo_ymin is not None:
        evaluator_kwargs["elo_ymin"] = elo_ymin
    plotter = TabArenaEvaluator(
        output_dir=output_dir,
        task_metadata=task_metadata,
        error_col=error_col,
        method_rename_map=method_rename_map,
        figure_file_type=figure_file_type,
        benchmark_name=benchmark_name,
        **evaluator_kwargs,
    )

    lb_df = plotter.eval(
        df_results=df_results,
        plot_extra_barplots=False,
        plot_times=True,
        calibration_framework=calibration_framework,
        average_seeds=average_seeds,
        leaderboard_kwargs=leaderboard_kwargs,
        **kwargs,
    )

    if add_dataset_count:
        lb_df["n_datasets_total"] = df_results["dataset"].nunique()

    return lb_df


def filter_to_valid_tasks(df_to_filter: pd.DataFrame, df_filter: pd.DataFrame) -> pd.DataFrame:
    dataset_fold_map = df_filter.groupby("dataset")["fold"].apply(set)

    def is_in(dataset: str, fold: int) -> bool:
        return (dataset in dataset_fold_map.index) and (fold in dataset_fold_map.loc[dataset])

    # filter `df_to_filter` to only the dataset, fold pairs that are present in `df_filter`
    is_in_lst = [
        is_in(dataset, fold)
        for dataset, fold in zip(
            df_to_filter["dataset"],
            df_to_filter["fold"],
            strict=False,
        )
    ]
    return df_to_filter[is_in_lst]


def prepare_data(
    df_results: pd.DataFrame,
    only_valid_tasks: str | list[str] | None = None,
    fillna: str | pd.DataFrame | None = None,
    remove_imputed: bool = False,
) -> pd.DataFrame:
    df_results = df_results.copy()

    if "method_type" not in df_results.columns:
        df_results["method_type"] = "baseline"
    if "method_subtype" not in df_results.columns:
        df_results["method_subtype"] = np.nan
    if "config_type" not in df_results.columns:
        df_results["config_type"] = np.nan
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False

    df_results["imputed"] = df_results["imputed"].fillna(0).astype(bool)

    if isinstance(only_valid_tasks, str):
        only_valid_tasks = [only_valid_tasks]
    if isinstance(only_valid_tasks, list):
        for filter_method in only_valid_tasks:
            # Filter to tasks present in a specific method
            df_filter = df_results[df_results["method"] == filter_method]
            if "imputed" in df_filter.columns:
                df_filter = df_filter[~df_filter["imputed"]]
            assert len(df_filter) != 0, (
                f"No method named '{filter_method}' remains to filter to!\n"
                f"Available tasks: {list(df_results['method'].unique())}"
            )
            df_results = filter_to_valid_tasks(
                df_to_filter=df_results,
                df_filter=df_filter,
            )

    if isinstance(fillna, str):
        fillna = df_results[df_results["method"] == fillna]
        if len(fillna) == 0:
            raise ValueError(
                "Missing fillna method in df_results!"
                f"\n\tfillna={fillna!r}"
                f"\n\tvalid_methods={sorted(df_results['method'].unique())}",
            )
    if fillna is not None:
        df_results = TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=fillna,
        )

    if remove_imputed:
        methods_imputed = df_results.groupby("method")["imputed"].sum()
        methods_imputed = list(methods_imputed[methods_imputed > 0].index)
        df_results = df_results[~df_results["method"].isin(methods_imputed)]

    return df_results


def _resolve_predicates(
    predicates: dict[str, Callable[[pd.DataFrame], pd.Series]] | None,
) -> dict[str, Callable[[pd.DataFrame], pd.Series]]:
    """Return the predicates dict to use, falling back to TabArenaContext's default."""
    if predicates is not None:
        return predicates
    return TabArenaContext.SUBSET_PREDICATES


def _evaluate_subset_expression(
    expression: str,
    df: pd.DataFrame,
    predicates: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.Series:
    """Evaluate a single subset expression against ``df`` and return a boolean mask.

    Supported syntax for one expression token:

    * ``"name"`` — atom; looks up ``name`` in ``predicates``.
    * ``"a|b"`` — union (OR) of atoms.
    * ``"!a"`` — negation; can be combined with union: ``"!a|b"`` is ``(NOT a) OR b``.

    Whitespace around tokens and operators is ignored.

    Parameters
    ----------
    predicates : dict[str, Callable] or None = None
        Mapping of subset name to predicate. If None, falls back to
        :attr:`TabArenaContext.SUBSET_PREDICATES`.
    """
    if not isinstance(expression, str):
        raise TypeError(f"Subset expression must be a string, got {type(expression).__name__}: {expression!r}")
    parts = [part.strip() for part in expression.split("|")]
    if any(not part for part in parts):
        raise ValueError(f"Empty atom in subset expression: {expression!r}")

    predicates = _resolve_predicates(predicates)

    mask: pd.Series | None = None
    for part in parts:
        negate = False
        if part.startswith("!"):
            negate = True
            part = part[1:].strip()
            if not part:
                raise ValueError(f"Dangling negation in subset expression: {expression!r}")
        if part not in predicates:
            valid = sorted(predicates)
            raise ValueError(f"Invalid subset name {part!r}. Valid names: {valid}")
        pred = predicates[part]
        # `SubsetPredicate` validates its `required_columns` against the grid (clear error on a
        # missing column); a plain callable is applied as-is for backward compatibility.
        sub = (pred.evaluate(df, name=part) if isinstance(pred, SubsetPredicate) else pred(df)).astype(bool)
        if negate:
            sub = ~sub
        mask = sub if mask is None else (mask | sub)
    return mask


def _task_grid_from_legacy_df(task_metadata: pd.DataFrame) -> pd.DataFrame:
    """Task-level grid (``dataset``/``fold``/``repeat``/``split`` + predicate columns) built from a
    legacy ``task_metadata`` DataFrame, by expanding the rectangular ``n_folds`` x ``n_repeats``
    grid per dataset. Mirrors :meth:`TaskMetadataCollection.task_grid` for the legacy input path
    (requires ``dataset``/``n_folds``/``n_repeats``; aliases ``n_samples_train_per_fold`` to
    ``max_train_rows``).
    """
    md = task_metadata.drop_duplicates("dataset").copy()
    if "max_train_rows" not in md.columns and "n_samples_train_per_fold" in md.columns:
        md["max_train_rows"] = md["n_samples_train_per_fold"]
    rows = []
    for r in md.to_dict("records"):
        n_folds, n_repeats = int(r["n_folds"]), int(r["n_repeats"])
        for repeat in range(n_repeats):
            for fold in range(n_folds):
                rows.append(
                    {
                        "dataset": r["dataset"],
                        "fold": fold,
                        "repeat": repeat,
                        "split": n_folds * repeat + fold,
                        "max_train_rows": r.get("max_train_rows"),
                        "n_features": r.get("n_features"),
                        "n_classes": r.get("n_classes"),
                        "problem_type": r.get("problem_type"),
                    },
                )
    return pd.DataFrame(
        rows,
        columns=["dataset", "fold", "repeat", "split", "max_train_rows", "n_features", "n_classes", "problem_type"],
    )


def _task_grid(task_metadata: pd.DataFrame | TaskMetadataCollection) -> pd.DataFrame:
    """The task-level grid the subset predicates evaluate against — native from a
    ``TaskMetadataCollection`` (:meth:`TaskMetadataCollection.task_grid`) or expanded from a
    legacy DataFrame (:func:`_task_grid_from_legacy_df`).
    """
    if isinstance(task_metadata, TaskMetadataCollection):
        return task_metadata.task_grid()
    return _task_grid_from_legacy_df(task_metadata)


def subset_tasks(
    df_results: pd.DataFrame,
    subset: list[str],
    tasks: list[tuple[str, int]] | None = None,
    folds: list[int] | None = None,
    datasets: list[str] | None = None,
    task_metadata_og: pd.DataFrame | TaskMetadataCollection | None = None,
    predicates: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Subset ``df_results`` by named predicates.

    The predicates are evaluated on the **task-level grid** derived from ``task_metadata_og``
    (one row per ``(dataset, fold, repeat, split)`` carrying the predicate metadata), never on
    ``df_results`` itself. The surviving ``(dataset, split)`` tasks are then used to filter
    ``df_results`` by a semi-join on ``(dataset, fold == split)`` (a results frame's ``fold`` is
    the split identifier). ``"lite"`` keys on the grid's ``split`` column.

    Each entry in ``subset`` is an expression. Items in the list are AND-ed together;
    within an item, ``|`` is a union (OR) and a leading ``!`` negates an atom. Examples:

    * ``["medium"]`` — datasets in the medium size bucket.
    * ``["medium|small"]`` — datasets that are medium OR small ("both medium and small").
    * ``["!medium"]`` — all except medium.
    * ``["classification", "!tiny"]`` — classification AND not tiny.

    Parameters
    ----------
    predicates : dict[str, Callable] or None = None
        Mapping of subset name to predicate. If None, falls back to
        :attr:`TabArenaContext.SUBSET_PREDICATES`. Callers with a
        ``TabArenaContext`` (or subclass) should pass ``context.subset_predicates``
        so context-specific subset definitions take effect.
    """
    from tabarena.benchmark.task.metadata import default_task_metadata_collection

    if task_metadata_og is None:
        task_metadata_og = default_task_metadata_collection()

    df_results = df_results.copy(deep=True)
    if subset:
        grid = _task_grid(task_metadata_og)
        for expression in subset:
            mask = _evaluate_subset_expression(expression, grid, predicates=predicates)
            grid = grid[mask.values]
        surviving = {(dataset, int(split)) for dataset, split in zip(grid["dataset"], grid["split"], strict=False)}
        keep = [
            (dataset, int(fold)) in surviving
            for dataset, fold in zip(df_results["dataset"], df_results["fold"], strict=False)
        ]
        df_results = df_results[pd.Series(keep, index=df_results.index)]

    if tasks is not None:
        task_index = pd.MultiIndex.from_tuples(tasks, names=["dataset", "fold"])
        df_results = df_results[pd.MultiIndex.from_frame(df_results[["dataset", "fold"]]).isin(task_index)]
    if datasets is not None:
        df_results = df_results[df_results["dataset"].isin(datasets)]
    if folds is not None:
        df_results = df_results[df_results["fold"].isin(folds)]
    return df_results.reset_index(drop=True)


def get_subsets_per_dataset(
    data_foundry_metadata: pd.DataFrame,
    predicates: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> dict[str, list[str]]:
    """For each dataset in ``data_foundry_metadata``, list the subset names
    (from ``predicates``) whose predicate the dataset satisfies.

    Predicates that reference columns not present in ``data_foundry_metadata``
    (e.g. ``"lite"`` needs a ``"split"`` column, ``"tabpfn"``/``"tabicl"`` need
    ``"n_features"``/``"n_classes"``) are skipped silently.

    Parameters
    ----------
    predicates : dict[str, Callable] or None = None
        Mapping of subset name to predicate. If None, falls back to
        :attr:`TabArenaContext.SUBSET_PREDICATES`.
    """
    predicates = _resolve_predicates(predicates)
    result: dict[str, list[str]] = {ds: [] for ds in data_foundry_metadata["dataset"].dropna().unique()}
    for subset_name, predicate in predicates.items():
        try:
            mask = predicate(data_foundry_metadata)
        except KeyError:
            continue
        qualifying = data_foundry_metadata.loc[mask, "dataset"].dropna()
        for ds in qualifying:
            result[ds].append(subset_name)
    return result


def subset_tasks_data_foundry(
    *,
    df_results: pd.DataFrame,
    subset: list[str],
    data_foundry_metadata: pd.DataFrame,
    predicates: dict[str, Callable[[pd.DataFrame], pd.Series]] | None = None,
) -> pd.DataFrame:
    """Subset ``df_results`` by filtering datasets according to ``data_foundry_metadata``.

    Supports the same expression syntax as :func:`subset_tasks` (``|`` for union,
    leading ``!`` for negation; list items are AND-ed).

    Parameters
    ----------
    predicates : dict[str, Callable] or None = None
        Mapping of subset name to predicate. If None, falls back to
        :attr:`TabArenaContext.SUBSET_PREDICATES`.
    """
    metadata = data_foundry_metadata
    for expression in subset:
        mask = _evaluate_subset_expression(expression, metadata, predicates=predicates)
        metadata = metadata[mask.values]

    valid_datasets = set(metadata["dataset"])
    df_results = df_results[df_results["dataset"].isin(valid_datasets)]

    # For each (method, dataset), drop the rows if the method did not run on every fold
    # that exists for that dataset (across all methods).
    n_folds_dataset = df_results.groupby("dataset")["fold"].transform("nunique")
    n_folds_method_dataset = df_results.groupby(["method", "dataset"])["fold"].transform("nunique")
    df_results = df_results[n_folds_method_dataset == n_folds_dataset]

    assert len(df_results) > 0, (
        "No results remain after subsetting! Please check the subset criteria and the data_foundry_metadata."
    )

    return df_results.reset_index(drop=True)
