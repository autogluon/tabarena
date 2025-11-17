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
    only_valid_tasks: bool = False,
    subset: str | list[str] | None = None,
    folds: list[int] | None = None,
    tabarena_context: TabArenaContext | None = None,
    fillna: str | pd.DataFrame | None = "RF (default)",
    score_on_val: bool = False,
    average_seeds: bool = True,
    tmp_treat_tasks_independently: bool = False,
    leaderboard_kwargs: dict | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    if tabarena_context is None:
        tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata

    paper_results = tabarena_context.load_results_paper(
        download_results="auto",
    )

    if new_results is not None:
        new_results = new_results.copy(deep=True)
        if "method_subtype" not in new_results:
            new_results["method_subtype"] = np.nan

        if only_valid_tasks:
            paper_results = filter_to_valid_tasks(
                df_to_filter=paper_results,
                df_filter=new_results,
            )

    if new_results is not None:
        df_results = pd.concat([paper_results, new_results], ignore_index=True)
    else:
        df_results = paper_results

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
        tmp_treat_tasks_independently=tmp_treat_tasks_independently,
        leaderboard_kwargs=leaderboard_kwargs,
    )


def compare(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    task_metadata: pd.DataFrame = None,
    calibration_framework: str | None = None,
    fillna: str | pd.DataFrame | None = None,
    score_on_val: bool = False,
    average_seeds: bool = True,
    tmp_treat_tasks_independently: bool = False,  # FIXME: Update
    leaderboard_kwargs: dict | None = None,
):
    df_results = df_results.copy()
    if "method_type" not in df_results:
        df_results["method_type"] = "baseline"
    if "method_subtype" not in df_results:
        df_results["method_subtype"] = np.nan
    if "config_type" not in df_results:
        df_results["config_type"] = None
    if "imputed" not in df_results:
        df_results["imputed"] = False

    if isinstance(fillna, str):
        fillna = df_results[df_results["method"] == fillna]
    if fillna is not None:
        df_results = TabArenaContext.fillna_metrics(
            df_to_fill=df_results,
            df_fillna=fillna,
        )

    if score_on_val:
        error_col = "metric_error_val"
        df_results = df_results[~df_results["metric_error_val"].isna()]
    else:
        error_col = "metric_error"

    imputed_names = get_imputed_names(df_results=df_results)

    baselines = list(
        df_results[
            df_results["method_type"].isin(['baseline', 'portfolio']) |
            ((df_results["method_type"] == "config") & df_results["method_subtype"].isna())
        ]["method"].unique()
    )

    plotter = TabArenaEvaluator(
        output_dir=output_dir,
        task_metadata=task_metadata,
        error_col=error_col,
    )

    return plotter.eval(
        df_results=df_results,
        baselines=baselines,
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


def get_imputed_names(df_results: pd.DataFrame) -> list[str]:
    # Handle imputation of names
    imputed_names = list(df_results["method"][df_results["imputed"] > 0].unique())
    if len(imputed_names) == 0:
        return []

    from tabarena.paper.paper_utils import get_method_rename_map

    # remove suffix
    imputed_names = [n.split(" (")[0] for n in imputed_names]
    imputed_names = [get_method_rename_map().get(n, n) for n in imputed_names]
    imputed_names = list(set(imputed_names))
    if "KNN" in imputed_names:
        imputed_names.remove("KNN")
    print(f"Model for which results were imputed: {imputed_names}")
    return imputed_names
