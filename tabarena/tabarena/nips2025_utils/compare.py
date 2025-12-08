from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.paper.tabarena_evaluator import TabArenaEvaluator


def compare_on_tabarena(
    output_dir: str | Path,
    new_results: pd.DataFrame | None = None,
    *,
    only_valid_tasks: bool | str | list[str] = False,
    subset: str | list[str] | None = None,
    folds: list[int] | None = None,
    tabarena_context: TabArenaContext | None = None,
    tabarena_context_kwargs: dict | None = None,
    fillna: str | pd.DataFrame | None = "RF (default)",
    score_on_val: bool = False,
    average_seeds: bool = True,
    remove_imputed: bool = False,
    tmp_treat_tasks_independently: bool = False,
    leaderboard_kwargs: dict | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if tabarena_context is None:
        if tabarena_context_kwargs is None:
            tabarena_context_kwargs = {}
        tabarena_context = TabArenaContext(**tabarena_context_kwargs)
    task_metadata = tabarena_context.task_metadata

    paper_results = tabarena_context.load_results_paper(
        download_results="auto",
    )

    if new_results is not None:
        new_results = new_results.copy(deep=True)
        if "method_subtype" not in new_results:
            new_results["method_subtype"] = np.nan

    if new_results is not None:
        df_results = pd.concat([paper_results, new_results], ignore_index=True)
    else:
        df_results = paper_results

    kwargs = {}
    if isinstance(only_valid_tasks, (str, list)):
        kwargs["only_valid_tasks"] = only_valid_tasks
    elif only_valid_tasks and new_results is not None:
        df_results = filter_to_valid_tasks(
            df_to_filter=df_results,
            df_filter=new_results,
        )

    if subset is not None or folds is not None:
        if subset is None:
            subset = []
        if isinstance(subset, str):
            subset = [subset]
        df_results = subset_tasks(df_results=df_results, subset=subset, folds=folds)

    return compare(
        df_results=df_results,
        output_dir=output_dir,
        task_metadata=task_metadata,
        fillna=fillna,
        calibration_framework=fillna,
        score_on_val=score_on_val,
        average_seeds=average_seeds,
        remove_imputed=remove_imputed,
        tmp_treat_tasks_independently=tmp_treat_tasks_independently,
        leaderboard_kwargs=leaderboard_kwargs,
        **kwargs,
    )


def compare(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    task_metadata: pd.DataFrame = None,
    only_valid_tasks: str | list[str] | None = None,
    calibration_framework: str | None = None,
    fillna: str | pd.DataFrame | None = None,
    score_on_val: bool = False,
    average_seeds: bool = True,
    tmp_treat_tasks_independently: bool = False,  # FIXME: Update
    leaderboard_kwargs: dict | None = None,
    remove_imputed: bool = False,
):
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
    if fillna is not None:
        df_results = TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=fillna,
        )

    if remove_imputed:
        methods_imputed = df_results.groupby("method")["imputed"].sum()
        methods_imputed = list(methods_imputed[methods_imputed > 0].index)
        df_results = df_results[~df_results["method"].isin(methods_imputed)]

    if score_on_val:
        error_col = "metric_error_val"
        df_results = df_results[~df_results["metric_error_val"].isna()]
    else:
        error_col = "metric_error"

    imputed_names = get_imputed_names(df_results=df_results)

    plotter = TabArenaEvaluator(
        output_dir=output_dir,
        task_metadata=task_metadata,
        error_col=error_col,
    )

    return plotter.eval(
        df_results=df_results,
        imputed_names=imputed_names,
        plot_extra_barplots=False,
        plot_times=True,
        plot_other=False,
        calibration_framework=calibration_framework,
        average_seeds=average_seeds,
        tmp_treat_tasks_independently=tmp_treat_tasks_independently,
        leaderboard_kwargs=leaderboard_kwargs,
    )


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


def subset_tasks(df_results: pd.DataFrame, subset: list[str], folds: list[int] = None) -> pd.DataFrame:
    from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

    df_results = df_results.copy(deep=True)
    for filter_subset in subset:
        if filter_subset == "classification":
            df_results = df_results[
                df_results["problem_type"].isin(["binary", "multiclass"])
            ]
        elif filter_subset == "binary":
            df_results = df_results[df_results["problem_type"] == "binary"]
        elif filter_subset == "multiclass":
            df_results = df_results[df_results["problem_type"] == "multiclass"]
        elif filter_subset == "regression":
            df_results = df_results[df_results["problem_type"] == "regression"]
        elif filter_subset == "medium+":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 10000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "medium":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 10000]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 250000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "small":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 10000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "tiny":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 2000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "tiny-small":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] < 10000]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 2000]
            valid_datasets = task_metadata["dataset"].unique()
            df_results = df_results[df_results["dataset"].isin(valid_datasets)]
        elif filter_subset == "lite":
            df_results = df_results[df_results["fold"] == 0]
        elif filter_subset == "tabicl":
            allowed_dataset = load_task_metadata(subset="TabICL")[
                "dataset"
            ].tolist()
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        elif filter_subset == "tabpfn":
            allowed_dataset = load_task_metadata(subset="TabPFNv2")[
                "dataset"
            ].tolist()
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        elif filter_subset == "tabpfn/tabicl":
            ad_tabicl = load_task_metadata(subset="TabICL")["dataset"].tolist()
            ad_tabpfn = load_task_metadata(subset="TabPFNv2")["dataset"].tolist()
            allowed_dataset = list(set(ad_tabicl).intersection(set(ad_tabpfn)))
            df_results = df_results[df_results["dataset"].isin(allowed_dataset)]
        else:
            raise ValueError(f"Invalid subset {subset} name!")

    if folds is not None:
        df_results = df_results[df_results["fold"].isin(folds)]
    df_results = df_results.reset_index(drop=True)
    return df_results


def get_imputed_names(df_results: pd.DataFrame, method_col="method") -> list[str]:
    # Handle imputation of names
    imputed_names = list(df_results[method_col][df_results["imputed"] > 0].unique())
    if len(imputed_names) == 0:
        return []

    from tabarena.paper.paper_utils import get_method_rename_map

    # remove suffix
    imputed_names = [n.split(" (")[0] for n in imputed_names]
    imputed_names = [get_method_rename_map().get(n, n) for n in imputed_names]
    imputed_names = list(set(imputed_names))
    print(f"Model for which results were imputed: {imputed_names}")
    return imputed_names
