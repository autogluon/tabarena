"""Meta-loss: a single scalar summarizing one method against a field of baselines.

The meta-loss normalizes each method's per-task error against the other methods
evaluated on the same task, then reduces those normalized metrics to one number with
a weighted geometric mean. Lower is better, and the value is only meaningful relative
to the field it was computed against: adding or removing baselines changes it.

Three normalizations run per error column, all computed per (task, seed) so that a
task with many seeds does not outvote a task with one:

- Improvability, `1 - best / error`: 0 when the method is the best on the task, and
  approaching 1 the further it is from the best.
- Rank, the percentile rank of the error among the methods on that task.
- Dominance gap, `(median_baseline - error) / (median_baseline - best_baseline)`,
  clipped to `[0, x_factor]`. Baseline statistics exclude the contender, so beating
  the best baseline scores above 1. It enters the geometric mean as the loss
  `x_factor - gap`, meaning a contender that is `x_factor` times better than the
  spread of the baselines gets a perfect score on this term.

An optional outlier metric penalizes catastrophic single-task failures, which the
across-task means above otherwise wash out. It z-scores the first error column's
per-task ranks against the non-outlier part of the distribution (everything up to Q3)
and keeps only the extreme tail, so tasks the method handles normally contribute
nothing to it.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Literal

import numpy as np
from scipy.stats import gmean, tmean, tstd

if TYPE_CHECKING:
    import pandas as pd

# Below this many tasks the outlier metric's Q3-based scaling is too noisy to be useful.
MIN_TASKS_FOR_OUTLIER_METRIC = 10


def compute_improvability(
    results: pd.DataFrame,
    error_col: str,
    groupby_cols: list[str],
) -> pd.Series:
    """Compute improvability, `1 - best_in_group / error`.

    Args:
        results: Per-row results to normalize.
        error_col: Column holding the error (lower is better, >= 0).
        groupby_cols: Columns defining the group the best error is taken over.

    Returns:
        Series aligned with `results`, where 0 means the row holds the group's best
        error and larger values mean more room for improvement.
    """
    best = results.groupby(groupby_cols, sort=False)[error_col].transform("min")
    return (1 - best / results[error_col]).fillna(0)


def compute_meta_loss(
    results_per_task: pd.DataFrame,
    contender: str,
    task_col: str = "task",
    method_col: str = "method",
    seed_col: str | None = None,
    error_col: str | list[str] = "error",
    error_weights: Literal["auto"] | list[float] | None = "auto",
    outlier_metric_weight: float | None = 1 / 5,
    dominance_gap_x_factor: int = 2,
    loss_eps: float = 1e-12,
) -> float:
    """Compute the meta-loss of one method against the other methods in `results_per_task`.

    Only (task, seed) combinations the contender has results for are scored; baselines
    may cover fewer of them.

    Args:
        results_per_task: Results of the contender and its baselines, one row per
            (task, seed, method). Errors must be finite and >= 0.
        contender: Value in `method_col` identifying the method to score.
        task_col: Column identifying tasks.
        method_col: Column identifying methods.
        seed_col: Optional column identifying repeats within a task. If None, each task
            is treated as having a single seed.
        error_col: Error column, or list of error columns to combine.
        error_weights: Geometric-mean weights for the error columns.
            - "auto" weights column `i` by `1 / (i + 1)`, so the first error column
              dominates and later ones act as tie-breakers.
            - None weights every error column equally.
            - A list must match the length of `error_col`.
        outlier_metric_weight: Weight of the outlier metric, or None to leave it out.
            It is dropped automatically for fewer than 10 tasks.
        dominance_gap_x_factor: Upper bound of the dominance gap, and the factor by
            which the contender must beat the baseline spread to score perfectly on it.
        loss_eps: Added to every error to keep near-zero errors from dominating the
            ratios, and used as a floor for degenerate standard deviations.

    Returns:
        The meta-loss. Lower is better.
    """
    error_cols = [error_col] if isinstance(error_col, str) else list(error_col)

    results = results_per_task.reset_index(drop=True).copy()
    if seed_col is None:
        seed_col = "__dummy_seed__"
        results[seed_col] = 0

    missing = [c for c in [task_col, method_col, seed_col, *error_cols] if c not in results]
    assert not missing, f"Columns missing from results_per_task: {missing}"
    assert isinstance(dominance_gap_x_factor, int), "dominance_gap_x_factor must be an int!"
    assert dominance_gap_x_factor > 1, "dominance_gap_x_factor must be greater than 1!"
    assert (results[method_col] == contender).any(), f"No rows for contender '{contender}' in the results!"

    # Restrict to what the contender was evaluated on, so baselines with extra tasks
    # cannot shift the normalization.
    results = results.merge(
        results.loc[results[method_col] == contender, [task_col, seed_col]].drop_duplicates(),
        on=[task_col, seed_col],
        how="inner",
    )
    assert (results[method_col] != contender).any(), "No baselines share tasks and seeds with the contender!"

    counts = results.groupby([task_col, seed_col, method_col], dropna=False).size()
    duplicated = counts[counts > 1]
    assert duplicated.empty, f"Duplicate (task, seed, method) rows detected. First few:\n{duplicated.head(20)}"
    assert np.isfinite(results[error_cols].to_numpy()).all(), "Error columns must not contain NaN or +-inf!"
    assert (results[error_cols].to_numpy() >= 0).all(), "Error columns must be >= 0!"

    if error_weights == "auto":
        error_weights = [1 / (i + 1) for i in range(len(error_cols))]
    elif error_weights is None:
        error_weights = [1.0] * len(error_cols)
    else:
        assert len(error_weights) == len(error_cols), "error_weights must match the number of error columns!"
    error_weights = dict(zip(error_cols, error_weights, strict=True))

    results[error_cols] = results[error_cols] + loss_eps
    experiment_cols = [task_col, seed_col]
    dominance_gap_name = f"{dominance_gap_x_factor}xDominanceGap"
    normalizations = ["Improvability", "Rank", dominance_gap_name]
    metrics = [f"{e}_{n}" for e in error_cols for n in normalizations]
    metric_weights = [error_weights[e] for e in error_cols for _ in normalizations]

    outlier_metric_col = None
    if outlier_metric_weight is not None and (
        outlier_metric_weight == 0 or results[task_col].nunique() < MIN_TASKS_FOR_OUTLIER_METRIC
    ):
        warnings.warn(
            "Dropping the outlier metric: its weight is zero or there are too few tasks.",
            UserWarning,
            stacklevel=2,
        )
        outlier_metric_weight = None
    if outlier_metric_weight is not None:
        outlier_metric_col = f"{error_cols[0]}_Rank"
        metrics += ["outlier_metric"]
        metric_weights += [outlier_metric_weight]

    # 1) Normalize each (task, seed) independently, before any averaging.
    for col in error_cols:
        results[f"{col}_Improvability"] = compute_improvability(results, col, experiment_cols)
        results[f"{col}_Rank"] = results.groupby(experiment_cols, sort=False)[col].rank(
            method="average", ascending=True, pct=True
        )

        # Baseline-only statistics. Masking the contender's errors to NaN lets the
        # NaN-skipping median/min broadcast the baseline statistic back onto every row,
        # the contender's included.
        baseline_error = results[col].where(results[method_col] != contender)
        baseline_groups = baseline_error.groupby([results[c] for c in experiment_cols], sort=False)
        median_error = baseline_groups.transform("median")
        min_error = baseline_groups.transform("min")
        results[f"{col}_{dominance_gap_name}"] = (
            (median_error - results[col]) / ((median_error - min_error) + loss_eps)
        ).clip(lower=0, upper=dominance_gap_x_factor)

    # 2) Average the contender's normalized metrics over seeds to get one row per task.
    results = results[results[method_col] == contender]
    per_task = results.groupby(task_col, sort=False)[[c for c in metrics if c != "outlier_metric"]].mean()

    if outlier_metric_col is not None:
        per_task["outlier_metric"] = _compute_outlier_metric(
            per_task[outlier_metric_col].to_numpy(dtype=float), loss_eps
        )

    # 3) Reduce across tasks, then across metrics.
    mean_per_metric = np.nanmean(per_task[metrics].to_numpy(dtype=float), axis=0)
    gap_idx = np.array([metrics.index(f"{e}_{dominance_gap_name}") for e in error_cols])
    if np.any(mean_per_metric[gap_idx] > dominance_gap_x_factor):
        warnings.warn(
            f"Mean {dominance_gap_name} exceeds {dominance_gap_x_factor}: the contender beats the baselines by "
            "more than the gap can express. Raise dominance_gap_x_factor for a more meaningful value.",
            UserWarning,
            stacklevel=2,
        )
    mean_per_metric[gap_idx] = np.clip(dominance_gap_x_factor - mean_per_metric[gap_idx], 0.0, None)

    # The eps keeps a perfect score on one metric from collapsing the whole product to zero.
    meta_loss = float(gmean(mean_per_metric + loss_eps, weights=np.asarray(metric_weights, dtype=float)))
    assert np.isfinite(meta_loss), "Meta-loss is not finite, this should not happen!"
    return meta_loss


def _compute_outlier_metric(rank_per_task: np.ndarray, loss_eps: float) -> np.ndarray:
    """Score only the tasks where the contender's rank is an extreme outlier.

    The scale is fit on the non-outlier part of the distribution (everything up to Q3),
    and every task at or below that threshold is set to NaN so it drops out of the
    across-task mean instead of diluting it.

    Args:
        rank_per_task: Per-task percentile ranks of the contender.
        loss_eps: Floor for a degenerate standard deviation.

    Returns:
        Array of the same length, NaN wherever the task is not an extreme outlier.
    """
    q3 = np.quantile(rank_per_task, 0.75)
    q3_mean = tmean(rank_per_task, limits=(None, q3))
    with warnings.catch_warnings():
        # A contender that ranks identically on every task makes the trimmed std
        # degenerate, which scipy reports as catastrophic cancellation. The
        # near-zero result is floored below, so the warning is noise.
        warnings.simplefilter("ignore", RuntimeWarning)
        q3_std = tstd(rank_per_task, limits=(None, q3))
    if np.isclose(q3_std, 0.0):
        q3_std = loss_eps

    z_scaled = (rank_per_task - q3_mean) / q3_std
    z_scaled[z_scaled <= (q3 - q3_mean) / q3_std] = np.nan

    # Keep at least a quarter of the tasks finite so the mean over them stays stable,
    # filling with the optimal value rather than inventing penalties.
    non_nan = ~np.isnan(z_scaled)
    expected_non_nan = int(np.ceil(0.25 * len(z_scaled)))
    if non_nan.sum() < expected_non_nan:
        fill_idx = np.flatnonzero(~non_nan)[: expected_non_nan - int(non_nan.sum())]
        z_scaled[fill_idx] = 1.0

    return z_scaled
