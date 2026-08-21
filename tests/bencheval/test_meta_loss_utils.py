from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from bencheval.meta_loss_utils import compute_improvability, compute_meta_loss

METHODS = ["contender", "baseline_a", "baseline_b", "baseline_c"]


def _results(n_tasks: int = 12, n_seeds: int = 1, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        [
            {"task": f"task_{t}", "seed": s, "method": method, "error": float(rng.random())}
            for t in range(n_tasks)
            for s in range(n_seeds)
            for method in METHODS
        ]
    )


def _with_error(results: pd.DataFrame, method: str, error: float, task: str | None = None) -> pd.DataFrame:
    results = results.copy()
    mask = results["method"] == method
    if task is not None:
        mask &= results["task"] == task
    results.loc[mask, "error"] = error
    return results


def test_improvability_is_zero_for_the_best_row_in_each_group():
    results = _results()
    improvability = compute_improvability(results, "error", ["task", "seed"])

    best_rows = results.groupby(["task", "seed"])["error"].idxmin()
    assert np.allclose(improvability.loc[best_rows], 0.0)
    assert (improvability.drop(best_rows) > 0).all()


def test_a_dominant_contender_scores_lower_than_a_dominated_one():
    results = _results()
    dominant = compute_meta_loss(_with_error(results, "contender", 0.0), contender="contender", seed_col="seed")
    dominated = compute_meta_loss(_with_error(results, "contender", 10.0), contender="contender", seed_col="seed")

    assert dominant < dominated


def test_only_the_contenders_tasks_are_scored():
    """Baselines evaluated on extra tasks must not shift the normalization."""
    results = _results()
    extra = results[results["task"] == "task_0"].copy()
    extra["task"] = "extra_task"
    extra = extra[extra["method"] != "contender"]

    scored = compute_meta_loss(results, contender="contender", seed_col="seed")
    with_extra = compute_meta_loss(pd.concat([results, extra]), contender="contender", seed_col="seed")

    assert scored == pytest.approx(with_extra)


def test_a_single_catastrophic_task_is_penalized_by_the_outlier_metric():
    """Which is the point of the metric: an across-task mean alone would wash this out."""
    results = _with_error(_results(n_tasks=20), "contender", 0.0)
    disaster = _with_error(results, "contender", 1e3, task="task_0")

    with_outlier = compute_meta_loss(disaster, contender="contender")
    without_outlier = compute_meta_loss(disaster, contender="contender", outlier_metric_weight=None)

    assert with_outlier > without_outlier


def test_the_outlier_metric_is_dropped_for_too_few_tasks():
    results = _results(n_tasks=5)
    with pytest.warns(UserWarning, match="too few tasks"):
        scored = compute_meta_loss(results, contender="contender", seed_col="seed")

    assert scored == pytest.approx(
        compute_meta_loss(results, contender="contender", seed_col="seed", outlier_metric_weight=None)
    )


def test_seeds_do_not_outvote_tasks():
    """A task's seeds are averaged into one row, so replicating a seed changes nothing."""
    results = _results(n_seeds=1)
    replicated = pd.concat([results, results.assign(seed=1)])

    assert compute_meta_loss(results, contender="contender", seed_col="seed") == pytest.approx(
        compute_meta_loss(replicated, contender="contender", seed_col="seed")
    )


def test_a_missing_seed_col_is_treated_as_one_seed_per_task():
    results = _results()

    assert compute_meta_loss(results.drop(columns="seed"), contender="contender") == pytest.approx(
        compute_meta_loss(results, contender="contender", seed_col="seed")
    )


def test_error_weights_shift_which_error_column_dominates():
    results = _results()
    results["error_2"] = 1 - results["error"]
    cols = ["error", "error_2"]

    auto = compute_meta_loss(results, contender="contender", seed_col="seed", error_col=cols)
    equal = compute_meta_loss(results, contender="contender", seed_col="seed", error_col=cols, error_weights=None)
    reversed_weights = compute_meta_loss(
        results, contender="contender", seed_col="seed", error_col=cols, error_weights=[0.5, 1.0]
    )

    assert auto != pytest.approx(equal)
    assert auto != pytest.approx(reversed_weights)


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r[r["method"] != "contender"], "No rows for contender"),
        (lambda r: r[r["method"] == "contender"], "No baselines"),
        (lambda r: pd.concat([r, r.head(1)]), "Duplicate"),
        (lambda r: r.assign(error=np.nan), "NaN"),
        (lambda r: r.assign(error=-1.0), ">= 0"),
    ],
)
def test_invalid_results_raise(mutate, match):
    with pytest.raises(AssertionError, match=match):
        compute_meta_loss(mutate(_results()), contender="contender", seed_col="seed")


def test_dominance_gap_x_factor_must_be_an_int_above_one():
    results = _results()
    with pytest.raises(AssertionError, match="must be an int"):
        compute_meta_loss(results, contender="contender", seed_col="seed", dominance_gap_x_factor=2.0)
    with pytest.raises(AssertionError, match="greater than 1"):
        compute_meta_loss(results, contender="contender", seed_col="seed", dominance_gap_x_factor=1)
