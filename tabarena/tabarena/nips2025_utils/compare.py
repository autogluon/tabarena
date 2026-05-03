from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator


def compare_on_tabarena(
    output_dir: str | Path,
    new_results: pd.DataFrame | None = None,
    ta_results: pd.DataFrame | None = None,
    *,
    only_valid_tasks: bool | str | list[str] = False,
    subset: str | list[str] | None = None,
    tasks: list[tuple[str, int]] | None = None,
    datasets: list[str] | None = None,
    folds: list[int] | None = None,
    tabarena_context: TabArenaContext | None = None,
    tabarena_context_kwargs: dict | None = None,
    fillna: str | pd.DataFrame | None = "RF (default)",
    calibration_framework: str | None = "auto",
    score_on_val: bool = False,
    average_seeds: bool = False,
    remove_imputed: bool = False,
    tmp_treat_tasks_independently: bool = False,
    leaderboard_kwargs: dict | None = None,
    **kwargs,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if tabarena_context is None:
        if tabarena_context_kwargs is None:
            tabarena_context_kwargs = {}
        tabarena_context = TabArenaContext(**tabarena_context_kwargs)
    task_metadata = tabarena_context.task_metadata

    # TODO: only methods that exist in runs
    #  Pair with (method, artifact_name)
    method_rename_map = tabarena_context.get_method_rename_map()

    if ta_results is None:
        ta_results = tabarena_context.load_results_paper(
            download_results="auto",
        )

    if new_results is not None:
        new_results = new_results.copy(deep=True)
        if "method_subtype" not in new_results.columns:
            new_results["method_subtype"] = np.nan

    if new_results is not None:
        df_results = pd.concat([ta_results, new_results], ignore_index=True)
    else:
        df_results = ta_results

    kwargs = kwargs.copy()
    if isinstance(only_valid_tasks, (str, list)):
        kwargs["only_valid_tasks"] = only_valid_tasks
    elif only_valid_tasks and new_results is not None:
        df_results = filter_to_valid_tasks(
            df_to_filter=df_results,
            df_filter=new_results,
        )

    return compare(
        df_results=df_results,
        output_dir=output_dir,
        task_metadata=task_metadata,
        subset=subset,
        tasks=tasks,
        datasets=datasets,
        folds=folds,
        fillna=fillna,
        calibration_framework=calibration_framework,
        score_on_val=score_on_val,
        average_seeds=average_seeds,
        remove_imputed=remove_imputed,
        tmp_treat_tasks_independently=tmp_treat_tasks_independently,
        leaderboard_kwargs=leaderboard_kwargs,
        method_rename_map=method_rename_map,
        **kwargs,
    )


def compare(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    task_metadata: pd.DataFrame = None,
    only_valid_tasks: str | list[str] | None = None,
    tasks: list[tuple[str, int]] | None = None,
    datasets: list[str] | None = None,
    calibration_framework: str | None = None,
    fillna: str | pd.DataFrame | None = None,
    score_on_val: bool = False,
    average_seeds: bool = False,
    tmp_treat_tasks_independently: bool = False,  # FIXME: Update
    leaderboard_kwargs: dict | None = None,
    remove_imputed: bool = False,
    method_rename_map: dict | None = None,
    figure_file_type: str = "pdf",
    add_dataset_count: bool = False,
    subset: list[str] | None = None,
    folds: list[int] | None = None,
    **kwargs,
):
    if subset is not None or folds is not None or datasets is not None or tasks is not None:
        if subset is None:
            subset = []
        if isinstance(subset, str):
            subset = [subset]
        df_results = subset_tasks(
            df_results=df_results,
            subset=subset,
            tasks=tasks,
            datasets=datasets,
            folds=folds,
            task_metadata_og=task_metadata,
        )

    df_results = prepare_data(
        df_results=df_results,
        only_valid_tasks=only_valid_tasks,
        fillna=fillna,
        remove_imputed=remove_imputed,
    )

    if datasets is not None:
        df_results = df_results[df_results["dataset"].isin(datasets)]

    if score_on_val:
        error_col = "metric_error_val"
        df_results = df_results[~df_results["metric_error_val"].isna()]
    else:
        error_col = "metric_error"

    if calibration_framework == "auto":
        if isinstance(fillna, pd.DataFrame):
            calibration_framework = None
        else:
            calibration_framework = fillna

    plotter = TabArenaEvaluator(
        output_dir=output_dir,
        task_metadata=task_metadata,
        error_col=error_col,
        method_rename_map=method_rename_map,
        figure_file_type=figure_file_type,
    )

    lb_df = plotter.eval(
        df_results=df_results,
        plot_extra_barplots=False,
        plot_times=True,
        calibration_framework=calibration_framework,
        average_seeds=average_seeds,
        tmp_treat_tasks_independently=tmp_treat_tasks_independently,
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
        is_in(dataset, fold) for dataset, fold in zip(
            df_to_filter["dataset"],
            df_to_filter["fold"],
        )]
    df_filtered = df_to_filter[is_in_lst]
    return df_filtered


def prepare_data(
    df_results: pd.DataFrame,
    only_valid_tasks: str | list[str] | None = None,
    fillna: str | pd.DataFrame | None = None,
    remove_imputed: bool = False,
) -> pd.DataFrame:
    df_results = df_results.copy()

    if isinstance(only_valid_tasks, str):
        only_valid_tasks = [only_valid_tasks]
    if isinstance(only_valid_tasks, list):
        for filter_method in only_valid_tasks:
            # Filter to tasks present in a specific method
            df_filter = df_results[df_results["method"] == filter_method]
            if "imputed" in df_filter.columns:
                df_filter = df_filter[df_filter["imputed"] != True]
            assert len(df_filter) != 0, \
                (f"No method named '{filter_method}' remains to filter to!\n"
                 f"Available tasks: {list(df_results['method'].unique())}")
            df_results = filter_to_valid_tasks(
                df_to_filter=df_results,
                df_filter=df_filter,
            )

    if "method_type" not in df_results.columns:
        df_results["method_type"] = "baseline"
    if "method_subtype" not in df_results.columns:
        df_results["method_subtype"] = np.nan
    if "config_type" not in df_results.columns:
        df_results["config_type"] = np.nan
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False

    if isinstance(fillna, str):
        fillna = df_results[df_results["method"] == fillna]
        if len(fillna) == 0:
            raise ValueError(
                "Missing fillna method in df_results!"
                f"\n\tfillna={fillna!r}"
                f"\n\tvalid_methods={sorted(list(df_results['method'].unique()))}"
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


_SUBSET_PREDICATES: dict[str, Callable[[pd.DataFrame], pd.Series]] = {
    # problem_type
    "binary": lambda df: df["problem_type"] == "binary",
    "multiclass": lambda df: df["problem_type"] == "multiclass",
    "classification": lambda df: df["problem_type"].isin(["binary", "multiclass"]),
    "regression": lambda df: df["problem_type"] == "regression",
    # size buckets keyed on training rows
    "large": lambda df: df["max_train_rows"].between(100_001, 1_000_350),  # +350 due to AMEX grouped dataset weird split
    "medium": lambda df: df["max_train_rows"].between(10_001, 100_000),
    "small": lambda df: df["max_train_rows"].between(1_001, 10_000),
    "tiny": lambda df: df["max_train_rows"].between(101, 1_000),
    # split / task type
    "random": lambda df: df["task_type"] == "random",
    "temporal": lambda df: df["task_type"] == "temporal",
    "grouped": lambda df: df["task_type"] == "grouped",
    # feature dimensionality / type
    "low-dim": lambda df: df["num_cols_after_preprocessing"] <= 100,
    "high-dim": lambda df: df["num_cols_after_preprocessing"] > 100,
    "text": lambda df: df["num_text_cols"] > 0,
    "high-cardinality": lambda df: df["num_high_cardinality_cats"] > 0,
    # foundation-model compatibility (operates on tabarena task_metadata columns)
    "tabpfn": lambda df: (df["max_train_rows"] <= 10_000) & (df["n_features"] <= 500) & (df["n_classes"] <= 10),
    "tabicl": lambda df: (df["max_train_rows"] <= 100_000) & (df["n_features"] <= 500) & (df["n_classes"] > 0),
    # row-level filter (requires a "fold" column; only meaningful when applied to df_results)
    "lite": lambda df: df["fold"] == 0,
}


def _evaluate_subset_expression(expression: str, df: pd.DataFrame) -> pd.Series:
    """Evaluate a single subset expression against ``df`` and return a boolean mask.

    Supported syntax for one expression token:

    * ``"name"`` — atom; looks up ``name`` in :data:`_SUBSET_PREDICATES`.
    * ``"a|b"`` — union (OR) of atoms.
    * ``"!a"`` — negation; can be combined with union: ``"!a|b"`` is ``(NOT a) OR b``.

    Whitespace around tokens and operators is ignored.
    """
    if not isinstance(expression, str):
        raise TypeError(f"Subset expression must be a string, got {type(expression).__name__}: {expression!r}")
    parts = [part.strip() for part in expression.split("|")]
    if any(not part for part in parts):
        raise ValueError(f"Empty atom in subset expression: {expression!r}")

    mask: pd.Series | None = None
    for part in parts:
        negate = False
        if part.startswith("!"):
            negate = True
            part = part[1:].strip()
            if not part:
                raise ValueError(f"Dangling negation in subset expression: {expression!r}")
        if part not in _SUBSET_PREDICATES:
            valid = sorted(_SUBSET_PREDICATES)
            raise ValueError(f"Invalid subset name {part!r}. Valid names: {valid}")
        sub = _SUBSET_PREDICATES[part](df).astype(bool)
        if negate:
            sub = ~sub
        mask = sub if mask is None else (mask | sub)
    return mask


def _join_results_with_metadata(
    df_results: pd.DataFrame,
    task_metadata: pd.DataFrame,
) -> pd.DataFrame:
    """Build a per-row view of ``df_results`` enriched with ``task_metadata`` columns.

    Aliases tabarena-style ``n_samples_train_per_fold`` to ``max_train_rows`` so the
    shared :data:`_SUBSET_PREDICATES` size predicates work uniformly across tabarena
    task_metadata and data-foundry metadata.
    """
    md = task_metadata.copy()
    if "max_train_rows" not in md.columns and "n_samples_train_per_fold" in md.columns:
        md["max_train_rows"] = md["n_samples_train_per_fold"]
    # Avoid clobbering columns already on df_results (e.g. problem_type from the run row).
    keep_cols = ["dataset"] + [c for c in md.columns if c != "dataset" and c not in df_results.columns]
    md = md[keep_cols]
    joined = df_results.merge(md, on="dataset", how="left")
    joined.index = df_results.index
    return joined


def subset_tasks(
    df_results: pd.DataFrame,
    subset: list[str],
    tasks: list[tuple[str, int]] | None = None,
    folds: list[int] | None = None,
    datasets: list[str] | None = None,
    task_metadata_og: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Subset ``df_results`` by named predicates from :data:`_SUBSET_PREDICATES`.

    Each entry in ``subset`` is an expression. Items in the list are AND-ed together;
    within an item, ``|`` is a union (OR) and a leading ``!`` negates an atom. Examples:

    * ``["medium"]`` — datasets in the medium size bucket.
    * ``["medium|small"]`` — datasets that are medium OR small ("both medium and small").
    * ``["!medium"]`` — all except medium.
    * ``["classification", "!tiny"]`` — classification AND not tiny.
    """
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata
    if task_metadata_og is None:
        task_metadata_og = load_task_metadata()

    df_results = df_results.copy(deep=True)
    if subset:
        joined = _join_results_with_metadata(df_results, task_metadata_og)
        for expression in subset:
            mask = _evaluate_subset_expression(expression, joined)
            df_results = df_results[mask.values]
            joined = joined[mask.values]

    if tasks is not None:
        task_index = pd.MultiIndex.from_tuples(tasks, names=["dataset", "fold"])
        df_results = df_results[
            pd.MultiIndex.from_frame(df_results[["dataset", "fold"]]).isin(task_index)
        ]
    if datasets is not None:
        df_results = df_results[df_results["dataset"].isin(datasets)]
    if folds is not None:
        df_results = df_results[df_results["fold"].isin(folds)]
    return df_results.reset_index(drop=True)


def get_subsets_per_dataset(data_foundry_metadata: pd.DataFrame) -> dict[str, list[str]]:
    """For each dataset in ``data_foundry_metadata``, list the subset names from
    :data:`_SUBSET_PREDICATES` whose predicate the dataset satisfies.

    Predicates that reference columns not present in ``data_foundry_metadata``
    (e.g. ``"lite"`` needs a ``"fold"`` column, ``"tabpfn"``/``"tabicl"`` need
    ``"n_features"``/``"n_classes"``) are skipped silently.
    """
    result: dict[str, list[str]] = {ds: [] for ds in data_foundry_metadata["dataset"].dropna().unique()}
    for subset_name, predicate in _SUBSET_PREDICATES.items():
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
) -> pd.DataFrame:
    """Subset ``df_results`` by filtering datasets according to ``data_foundry_metadata``.

    Supports the same expression syntax as :func:`subset_tasks` (``|`` for union,
    leading ``!`` for negation; list items are AND-ed).
    """
    metadata = data_foundry_metadata
    for expression in subset:
        mask = _evaluate_subset_expression(expression, metadata)
        metadata = metadata[mask.values]

    valid_datasets = set(metadata["dataset"])
    df_results = df_results[df_results["dataset"].isin(valid_datasets)]

    # For each (method, dataset), drop the rows if the method did not run on every fold
    # that exists for that dataset (across all methods).
    n_folds_dataset = df_results.groupby("dataset")["fold"].transform("nunique")
    n_folds_method_dataset = df_results.groupby(["method", "dataset"])["fold"].transform("nunique")
    df_results = df_results[n_folds_method_dataset == n_folds_dataset]

    assert len(df_results) > 0, "No results remain after subsetting! Please check the subset criteria and the data_foundry_metadata."

    return df_results.reset_index(drop=True)