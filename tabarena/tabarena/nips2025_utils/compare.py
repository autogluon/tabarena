from __future__ import annotations

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
    datasets: list[str] | None = None,
    folds: list[int] | None = None,
    tabarena_context: TabArenaContext | None = None,
    tabarena_context_kwargs: dict | None = None,
    fillna: str | pd.DataFrame | None = "RF (default)",
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

    if subset is not None or folds is not None or datasets is not None:
        if subset is None:
            subset = []
        if isinstance(subset, str):
            subset = [subset]
        df_results = subset_tasks(df_results=df_results, subset=subset, folds=folds, datasets=datasets)

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
        method_rename_map=method_rename_map,
        **kwargs,
    )


def compare(
    df_results: pd.DataFrame,
    output_dir: str | Path,
    task_metadata: pd.DataFrame = None,
    only_valid_tasks: str | list[str] | None = None,
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
    **kwargs,
):
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


def subset_tasks(
    df_results: pd.DataFrame,
    subset: list[str],
    folds: list[int] = None,
    datasets: list[str] = None,
) -> pd.DataFrame:
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
        elif filter_subset == "small+":
            task_metadata = load_task_metadata()
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] >= 2000]
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

    if datasets is not None:
        df_results = df_results[df_results["dataset"].isin(datasets)]
    if folds is not None:
        df_results = df_results[df_results["fold"].isin(folds)]
    df_results = df_results.reset_index(drop=True)
    return df_results


# TODO:
#  - import from TabArena once we are able to pull/rebase
#  - make this something stored in the metadata...
def get_split_idx(
    fold: int = 0,
    repeat: int = 0,
    sample: int = 0,
    n_folds: int = 1,
    n_repeats: int = 1,
    n_samples: int = 1,
) -> int:
    assert fold < n_folds
    assert repeat < n_repeats
    assert sample < n_samples
    split_idx = n_folds * n_samples * repeat + n_samples * fold + sample
    return split_idx

def _add_split_idx_to_task_metadata(task_metadata: pd.DataFrame) -> pd.DataFrame:
    n_folds_per_row = task_metadata.groupby("dataset")["fold"].transform("max") + 1
    n_repeats_per_row = task_metadata.groupby("dataset")["repeat"].transform("max") + 1

    task_metadata["legacy_split_idx"] = [
        get_split_idx(fold=fold, repeat=repeat, n_folds=n_folds, n_repeats=n_repeats)
        for fold, repeat, n_folds, n_repeats in zip(
            task_metadata["fold"],
            task_metadata["repeat"],
            n_folds_per_row,
            n_repeats_per_row,
        )
    ]
    return task_metadata

def df_results_from_task_metadata_slice(*, df_results: pd.DataFrame, task_metadata_slice: pd.DataFrame) -> pd.DataFrame:
    valid_pairs = set(
        zip(task_metadata_slice["dataset"], task_metadata_slice["legacy_split_idx"])
    )
    mask = [
        (dataset, fold) in valid_pairs
        for dataset, fold in zip(df_results["dataset"], df_results["fold"])
    ]
    return df_results[mask].reset_index(drop=True)

def subset_tasks_new(
    *,
    df_results: pd.DataFrame,
    subset: list[str],
    task_metadata: pd.DataFrame,
) -> pd.DataFrame:
    df_results = df_results.copy(deep=True)
    task_metadata = task_metadata.copy(deep=True)
    task_metadata = _add_split_idx_to_task_metadata(task_metadata)

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
        elif filter_subset == "large":
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] > 100_000]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 1_000_000]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "medium":
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] > 10_000]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 100_000]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "small":
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 10_000]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] > 1_000]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "tiny":
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] > 100]
            task_metadata = task_metadata[task_metadata["n_samples_train_per_fold"] <= 1_000]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "random":
            task_metadata = task_metadata[task_metadata["time_on"].isna() & task_metadata["group_on"].isna()]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "temporal":
            task_metadata = task_metadata[~task_metadata["time_on"].isna()]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "grouped":
            task_metadata = task_metadata[~task_metadata["group_on"].isna()]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "low-dim":
            task_metadata = task_metadata[task_metadata["num_features_train"] <= 100]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "high-dim":
            task_metadata = task_metadata[task_metadata["num_features_train"] > 100]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "text":
            task_metadata = task_metadata[task_metadata["has_text"]]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "high-cardinality":
            task_metadata = task_metadata[task_metadata["has_high_cardinality_categorical"]]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        elif filter_subset == "binary-features":
            task_metadata = task_metadata[task_metadata["has_binary"]]
            df_results = df_results_from_task_metadata_slice(df_results=df_results, task_metadata_slice=task_metadata)
        else:
            raise ValueError(f"Invalid subset {subset} name!")

    return df_results.reset_index(drop=True)
