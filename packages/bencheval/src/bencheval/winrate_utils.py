from __future__ import annotations

import numpy as np
import pandas as pd


def compute_winrate_matrix(
    results_per_task: pd.DataFrame,
    task_col: str | list[str] = "task",
    method_col: str = "method",
    error_col: str = "error",
    seed_col: str | None = None,
    tie_decimals: int | None = None,
) -> pd.DataFrame:
    """Pairwise win-rate matrix with optional seeding.

    - If seed_col is provided:
        * Compare methods only within the same (task, seed).
        * Each task contributes equally, regardless of #seeds.
    - If seed_col is None:
        * Treat each task as having a single dummy seed.
    - If tie_decimals is set:
        * Round error to this many decimals before comparison so two methods
          that hit the same metric via different code paths (floating-point
          noise in the last few bits) still count as tied. Default None
          preserves the original exact-equality behavior.

    Returns:
    -------
    pd.DataFrame
        Square DataFrame [methods x methods], (i,j) = win-rate of i vs j.
        Diagonal is NaN.
    """
    # If no seed_col, create a dummy one so logic is unified
    if seed_col is None:
        seed_col = "__dummy_seed__"
        results_per_task = results_per_task.copy()
        results_per_task[seed_col] = 0

    if tie_decimals is not None:
        results_per_task = results_per_task.copy()
        results_per_task[error_col] = results_per_task[error_col].round(tie_decimals)

    methods = pd.Index(results_per_task[method_col].unique())
    midx = {m: i for i, m in enumerate(methods)}
    n = len(methods)

    wins_global = np.zeros((n, n), dtype=np.float32)
    matches_global = np.zeros((n, n), dtype=np.float32)

    # Group by task
    for _task, df_task in results_per_task.groupby(task_col, sort=False):
        wins_task = np.zeros((n, n), dtype=np.float32)
        matches_task = np.zeros((n, n), dtype=np.float32)

        seeds = df_task[seed_col].unique()
        n_seeds = len(seeds)

        for _, g in df_task.groupby(seed_col, sort=False):
            idx = g[method_col].map(midx).to_numpy()
            err = g[error_col].to_numpy()

            # this logic ensures infinity values are handled correctly
            err_i = err[:, None]
            err_j = err[None, :]

            # win: err_i < err_j
            wins = (err_i < err_j).astype(np.float32)
            # tie: err_i == err_j
            ties = (err_i == err_j).astype(np.float32)

            block_wins = wins + 0.5 * ties
            np.fill_diagonal(block_wins, 0.0)

            block_matches = np.ones((len(idx), len(idx)), dtype=np.float32)
            np.fill_diagonal(block_matches, 0.0)

            ix = np.ix_(idx, idx)
            wins_task[ix] += block_wins
            matches_task[ix] += block_matches

        # Normalize so each task contributes equally
        if n_seeds > 0:
            wins_task /= n_seeds
            matches_task /= n_seeds

        wins_global += wins_task
        matches_global += matches_task

    with np.errstate(divide="ignore", invalid="ignore"):
        win_rates = wins_global / matches_global
    np.fill_diagonal(win_rates, np.nan)

    # Sort by average win-rate
    avg_wr = np.nanmean(win_rates, axis=1)
    order = np.argsort(-avg_wr)

    win_rates_df = pd.DataFrame(win_rates, index=methods, columns=methods)
    return win_rates_df.iloc[order, order]


def compute_winrate(
    results_per_task: pd.DataFrame,
    task_col: str | list[str] = "task",
    method_col: str = "method",
    error_col: str = "error",
    seed_col: str | None = None,
    sort_desc: bool = True,
    tie_decimals: int | None = None,
) -> pd.Series:
    """Average win-rate per method.

    This calls `compute_winrate_matrix` and then averages across opponents.
    Keeps identical behavior with respect to seeding and task weighting.

    Parameters
    ----------
    results_per_task : pd.DataFrame
        Must contain task, method, and error columns.
    task_col : str or list[str], default="task"
        Column(s) identifying tasks.
    method_col : str, default="method"
        Column identifying methods.
    error_col : str, default="error"
        Column with error values.
    seed_col : str or None, default=None
        Column indicating random seeds for tasks. If None, tasks are treated
        as having a single dummy seed.
    sort_desc : bool, default=True
        If True, return methods sorted from highest to lowest avg win-rate.

    Returns:
    -------
    pd.Series
        Index = methods, Values = average win-rate.
    """
    winrate_matrix = compute_winrate_matrix(
        results_per_task=results_per_task,
        task_col=task_col,
        method_col=method_col,
        error_col=error_col,
        seed_col=seed_col,
        tie_decimals=tie_decimals,
    )

    avg_wr = winrate_matrix.mean(axis=1, skipna=True)
    avg_wr.name = "winrate"
    avg_wr.index.name = method_col
    return avg_wr.sort_values(ascending=False) if sort_desc else avg_wr
