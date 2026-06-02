from __future__ import annotations

import numpy as np
import pandas as pd


def compute_batched_tasks_all(batch_size_groups: list[int], time_per_dataset: pd.DataFrame, methods: list, folds: list) -> list[list[tuple]]:
    """

    Parameters
    ----------
    batch_size_groups
    time_per_dataset
    methods
    folds

    Returns
    -------

    """
    tasks_all = []

    for batch_size in batch_size_groups:
        df_datasets_in_batch_size = time_per_dataset[time_per_dataset["batch_size_group"] == batch_size]
        datasets_in_batch_size = list(df_datasets_in_batch_size.index)
        tasks_in_ = [(dataset, fold, method) for dataset in datasets_in_batch_size for fold in folds for method in methods]
        tasks_in_batched = []
        n_tasks = len(tasks_in_)
        prev_i = 0
        for i in range(batch_size, n_tasks, batch_size):
            tasks_in_batched.append(tasks_in_[prev_i:i])
            prev_i = i
        if prev_i < n_tasks:
            tasks_in_batched.append(tasks_in_[prev_i:])

        assert sum([len(t) for t in tasks_in_batched]) == len(tasks_in_)

        if len(tasks_in_batched) > 0:
            tasks_all.append(tasks_in_batched)

    tasks_all = [task for tasks_all_batch in tasks_all for task in tasks_all_batch]

    return tasks_all


def estimate_benchmark_runtime(n_methods, n_folds, time_per_dataset: pd.DataFrame, target_time: float, time_per_instance_startup: float):
    """
    Estimate the total runtime of a benchmark run

    TODO: Add max_parallel_instances

    Parameters
    ----------
    n_methods
    n_folds
    time_per_dataset
    target_time
    time_per_instance_startup

    Returns
    -------

    """
    n_datasets_per_group = time_per_dataset.value_counts("batch_size_group").to_frame().reset_index(drop=False)
    n_datasets_per_group["n_instances_group"] = n_methods * n_datasets_per_group["count"] / n_datasets_per_group["batch_size_group"]
    n_datasets_per_group["n_instances_group"] = np.ceil(n_datasets_per_group["n_instances_group"]).astype(int)

    n_datasets = len(time_per_dataset)
    n_runs = n_methods * n_datasets * n_folds

    total_instances = n_datasets_per_group["n_instances_group"].sum()

    instance_startup_time_total = total_instances * time_per_instance_startup

    total_time_s = instance_startup_time_total + target_time
    total_time_hr = total_time_s / 3600
    print(f"{total_time_hr:.2f} hours to complete {n_runs} runs ({n_datasets} datasets, {n_folds} folds, {n_methods} methods)")
    return total_time_s


def get_time_per_dataset_info(metrics: pd.DataFrame) -> pd.DataFrame:
    """

    Parameters
    ----------
    metrics:
        Output of `Evaluator(repo).compare_metrics(fillna=False).reset_index(drop=False)`

    Returns
    -------

    """
    dataset_count = metrics["dataset"].value_counts()

    time_per_dataset = metrics.groupby("dataset")[["time_train_s", "time_infer_s"]].sum()
    time_per_dataset["time_total_s"] = time_per_dataset["time_train_s"] + time_per_dataset["time_infer_s"]
    time_per_dataset["time_total_frac"] = time_per_dataset["time_total_s"] / time_per_dataset["time_total_s"].sum()

    dataset_count_ref = time_per_dataset.index.map(dataset_count)

    time_per_dataset["time_total_s_mean"] = time_per_dataset["time_total_s"] / dataset_count_ref

    time_per_dataset_q90 = metrics.groupby("dataset")[["time_train_s", "time_infer_s"]].quantile(q=0.9) * metrics.groupby("dataset")[
        ["time_train_s", "time_infer_s"]].count()
    time_per_dataset_q90["time_total_s"] = time_per_dataset_q90["time_train_s"] + time_per_dataset_q90["time_infer_s"]
    time_per_dataset_q90["time_total_s_q90"] = time_per_dataset_q90["time_total_s"] / dataset_count_ref

    time_per_dataset["time_total_s_q90"] = time_per_dataset_q90["time_total_s_q90"]
    return time_per_dataset
