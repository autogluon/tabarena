from __future__ import annotations
import pandas as pd
import numpy as np


def compute_weighted_mean_by_task(
    df: pd.DataFrame,
    value_col: str,
    task_col: str | list[str] = "task",
    seed_col: str | None = None,
    method_col: str = "method",
    weight_col: str | None = None,
    sort_asc: bool = True,
) -> pd.Series:
    """
    Compute the equal-task-weighted mean of a column for each method.

    - If seed_col is provided:
        * Aggregate values per (task, seed, method) first.
        * Average over seeds within each task so every task contributes equally.
    - If seed_col is None:
        * Treat each task as having a single dummy seed.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing task, method, and value columns.
    value_col : str
        Name of the column to average.
    task_col : str or list[str], default="task"
        Column(s) identifying tasks.
    seed_col : str or None, default=None
        Optional seed column for multiple runs per task.
    method_col : str, default="method"
        Column identifying methods.
    weight_col : str or None, default=None
        Optional column of numeric weights to compute a weighted mean within groups.
    sort_asc : bool, default=True
        Whether to sort output in ascending order (True) or descending order (False)

    Returns
    -------
    pd.Series
        Index = methods, Values = equal-task-weighted mean of `value_col`.
    """
    df = df.copy()

    if not isinstance(task_col, list):
        task_col = [task_col]

    group_task = [*task_col, method_col]
    if seed_col is not None:
        group_task_seed = group_task + [seed_col]
    else:
        group_task_seed = group_task

    # Step 1: Aggregate to per-(task, seed, method)
    if weight_col is not None:
        agg_df = (
            df.groupby(group_task_seed, sort=False)
            .apply(lambda g: np.average(g[value_col], weights=g[weight_col]))
            .reset_index(name=value_col)
        )
    else:
        agg_df = (
            df.groupby(group_task_seed, sort=False)[value_col]
            .mean()
            .reset_index()
        )

    # Step 2: Average over seeds within each (task, method)
    task_avg = agg_df.groupby(group_task, sort=False)[value_col].mean().reset_index()

    # Step 3: Average across tasks equally per method
    mean_per_method = task_avg.groupby(method_col, sort=False)[value_col].mean().sort_values(ascending=sort_asc)

    return mean_per_method
