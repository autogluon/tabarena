from __future__ import annotations

import numpy as np
import pandas as pd

from tabarena.utils.normalized_scorer import NormalizedScorer

dataset_col = "dataset"
metric_col = "metric"
framework_col = "framework"

df_results_by_dataset = pd.DataFrame(
    [
        ["dataset1", "xgboost1", 1.0],
        ["dataset1", "xgboost3", 3.0],
        ["dataset1", "xgboost2", 2.0],
        ["dataset2", "xgboost1", 10.0],
        ["dataset2", "xgboost3", 30.0],
        ["dataset2", "xgboost2", 20.0],
    ],
    columns=[dataset_col, framework_col, metric_col],
)


def test_normalized_scorer():
    scorer = NormalizedScorer(
        df_results=df_results_by_dataset,
        tasks=["dataset1", "dataset2"],
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
    )
    query_expected = [
        (1.0, 0.0),
        (2.0, 1.0),
        (1.5, 0.5),
        (3.0, 1.0),
        (0.0, 0.0),
    ]
    for query, expected in query_expected:
        print(scorer.rank("dataset1", query))
        assert np.isclose(scorer.rank("dataset1", query), expected)


def _scorer() -> NormalizedScorer:
    return NormalizedScorer(
        df_results_by_dataset,
        tasks=list(df_results_by_dataset[dataset_col].unique()),
        baseline=None,
        metric_error_col=metric_col,
        task_col=dataset_col,
        framework_col=framework_col,
    )


def test_rank_many_matches_rank_element_for_element():
    scorer = _scorer()
    tasks = pd.Index(df_results_by_dataset[dataset_col])
    errors = df_results_by_dataset[metric_col].to_numpy()

    scalar = np.array([scorer.rank(task=task, error=error) for task, error in zip(tasks, errors, strict=False)])

    assert np.array_equal(scorer.rank_many(tasks=tasks, errors=errors), scalar)


def test_rank_many_returns_nan_for_a_task_the_scorer_never_saw():
    """`get_indexer` reports -1 for an absent key, which would otherwise index from the array's end."""
    scorer = _scorer()
    tasks = pd.Index(["dataset1", "never-seen", "dataset2"])

    scores = scorer.rank_many(tasks=tasks, errors=np.array([1.0, 1.0, 10.0]))

    assert np.isnan(scores[1])
    assert not np.isnan(scores[[0, 2]]).any()


def test_rank_many_handles_a_multiindex_task_key():
    df = df_results_by_dataset.assign(fold=[0, 0, 0, 1, 1, 1])
    scorer = NormalizedScorer(
        df,
        tasks=[tuple(task) for task in df[[dataset_col, "fold"]].drop_duplicates().to_numpy().tolist()],
        baseline=None,
        metric_error_col=metric_col,
        task_col=[dataset_col, "fold"],
        framework_col=framework_col,
    )
    tasks = pd.MultiIndex.from_arrays([df[dataset_col], df["fold"]])

    scores = scorer.rank_many(tasks=tasks, errors=df[metric_col].to_numpy())

    expected = np.array(
        [scorer.rank(task=task, error=error) for task, error in zip(tasks, df[metric_col], strict=False)]
    )
    assert np.array_equal(scores, expected)
