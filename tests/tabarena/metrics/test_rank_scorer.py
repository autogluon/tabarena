from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.utils.rank_utils import RankScorer

dataset_col = "dataset"
metric_col = "metric"
framework_col = "framework"

df_results_by_dataset = pd.DataFrame(
    [
        ["task1", "xgboost1", 1.0],
        ["task1", "xgboost3", 3.0],
        ["task1", "xgboost2", 2.0],
        ["task2", "xgboost1", 10.0],
        ["task2", "xgboost3", 30.0],
        ["task2", "xgboost2", 20.0],
    ],
    columns=[dataset_col, framework_col, metric_col],
)


def test_rank_scorer():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        pct=False,
        include_partial=True,
        ties_win=False,
    )
    query_expected = [
        (0.0, 0.0),
        (0.8, 0.4),
        (1.0, 0.5),
        (1.5, 1.25),
        (2.0, 1.5),
        (4.0, 3.16666666),
        (8.0, 3.5),
    ]
    for query, expected in query_expected:
        assert np.isclose(rank_scorer.rank("task1", query), expected)


def test_rank_scorer_pct():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        pct=True,
        include_partial=True,
        ties_win=False,
    )
    query_expected = [
        (0.0, 0.0),
        (0.8, 0.1142857142857143),
        (1.0, 0.14285714285714285),
        (1.5, 0.35714285714285715),
        (2.0, 0.42857142857142855),
        (2.5, 0.6428571428571429),
        (3.0, 0.7142857142857143),
        (4.0, 0.9047619047619048),
        (8.0, 1.0),
    ]
    for query, expected in query_expected:
        assert np.isclose(rank_scorer.rank("task1", query), expected)


def test_rank_scorer_ties_win():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        ties_win=True,
        include_partial=False,
        pct=False,
    )
    query_expected = [
        (0.0, 0),
        (0.8, 0),
        (1.0, 0),
        (1.5, 1),
        (2.0, 1),
        (4.0, 3),
        (8.0, 3),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("task1", query) == expected


def test_rank_scorer_pct_ties_win():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        ties_win=True,
        include_partial=False,
        pct=True,
    )
    query_expected = [
        (0.0, 0.0),
        (0.8, 0.0),
        (1.0, 0.0),
        (1.5, 1 / 3),
        (2.0, 1 / 3),
        (3.0, 2 / 3),
        (4.0, 1.0),
        (8.0, 1.0),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("task1", query) == expected


def test_rank_scorer_not_partial():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        pct=False,
        include_partial=False,
        ties_win=False,
    )
    query_expected = [
        (0.0, 0.0),
        (0.8, 0.0),
        (1.0, 0.5),
        (1.5, 1),
        (2.0, 1.5),
        (4.0, 3),
        (8.0, 3),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("task1", query) == expected


def test_rank_scorer_pct_not_partial():
    rank_scorer = RankScorer(
        df_results=df_results_by_dataset,
        tasks=["task1", "task2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
        pct=True,
        include_partial=False,
        ties_win=False,
    )
    query_expected = [
        (0.0, 0.0),
        (0.8, 0.0),
        (1.0, 1 / 6),
        (1.5, 1 / 3),
        (2.0, 1 / 2),
        (2.5, 2 / 3),
        (3.0, 5 / 6),
        (4.0, 1.0),
        (8.0, 1.0),
    ]
    for query, expected in query_expected:
        assert rank_scorer.rank("task1", query) == expected


@pytest.mark.parametrize(
    ("ties_win", "include_partial", "pct"),
    [(False, True, False), (False, True, True), (False, False, False), (True, False, False), (True, False, True)],
)
def test_rank_many_matches_rank(ties_win, include_partial, pct):
    """``rank_many`` reproduces per-row ``rank`` on ties, zeros, empty tasks, out-of-range and NaN errors."""
    rng = np.random.default_rng(0)
    rows = []
    for t in range(30):
        base = np.round(rng.random(rng.integers(0, 8)) * 3, 1)  # empty lists, ties and zeros included
        if t % 5 == 0:
            base = np.append(base, 0.0)
        worst = base.max() * 2 if len(base) else 1.0
        errors = np.concatenate([base, base + 0.05, [0.0, 10.0, np.nan, worst]])
        rows += [{"task": f"t{t}", "framework": f"m{j}", "metric_error": e} for j, e in enumerate(errors)]
    df = pd.DataFrame(rows)
    df_results = df.dropna().groupby(["task", "framework"]).first().reset_index()
    scorer = RankScorer(
        df_results=df_results,
        tasks=sorted(df["task"].unique()),
        ties_win=ties_win,
        include_partial=include_partial,
        pct=pct,
    )
    expected = np.array([scorer.rank(task, error) for task, error in zip(df["task"], df["metric_error"], strict=True)])
    actual = scorer.rank_many(tasks=df["task"].to_numpy(), errors=df["metric_error"].to_numpy())
    np.testing.assert_array_equal(actual, expected)
