from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.compare import subset_tasks


def _legacy_df() -> pd.DataFrame:
    """Two datasets, each n_folds=2 x n_repeats=1 (splits 0,1):
    small_ds (binary, ~100 train rows) and big_ds (regression, 50k train rows).
    """
    return pd.DataFrame(
        {
            "dataset": ["small_ds", "big_ds"],
            "name": ["small_ds", "big_ds"],
            "tid": [0, 1],
            "problem_type": ["binary", "regression"],
            "n_folds": [2, 2],
            "n_repeats": [1, 1],
            "n_features": [10, 10],
            "n_classes": [2, 0],
            "NumberOfInstances": [150, 75_000],
            "n_samples_train_per_fold": [100, 50_000],
            "n_samples_test_per_fold": [50, 25_000],
            "target_feature": ["t", "t"],
        }
    )


def _df_results() -> pd.DataFrame:
    # One row per (dataset, fold) — fold is the split identifier (0, 1).
    return pd.DataFrame(
        {
            "dataset": ["small_ds", "small_ds", "big_ds", "big_ds"],
            "fold": [0, 1, 0, 1],
            "method": ["m", "m", "m", "m"],
            "metric_error": [0.1, 0.2, 0.3, 0.4],
        }
    )


def _kept(df: pd.DataFrame) -> set[tuple[str, int]]:
    return {(d, int(f)) for d, f in zip(df["dataset"], df["fold"], strict=False)}


# (subset, expected surviving (dataset, fold) set)
_CASES = [
    (["small"], {("small_ds", 0), ("small_ds", 1)}),
    (["regression"], {("big_ds", 0), ("big_ds", 1)}),
    (["classification"], {("small_ds", 0), ("small_ds", 1)}),
    (["lite"], {("small_ds", 0), ("big_ds", 0)}),  # split == 0
    (["small", "lite"], {("small_ds", 0)}),  # AND
    (["!small"], {("big_ds", 0), ("big_ds", 1)}),  # negation
    (["small|lite"], {("small_ds", 0), ("small_ds", 1), ("big_ds", 0)}),  # mixed OR (dataset-level | split-level)
]


class TestSubsetTasksTaskLevel:
    @pytest.mark.parametrize(("subset", "expected"), _CASES)
    def test_collection_input(self, subset, expected):
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())
        out = subset_tasks(df_results=_df_results(), subset=subset, task_metadata_og=coll)
        assert _kept(out) == expected

    @pytest.mark.parametrize(("subset", "expected"), _CASES)
    def test_legacy_df_input_matches_collection(self, subset, expected):
        # The legacy-DataFrame grid path must produce the same result as the native collection path.
        out = subset_tasks(df_results=_df_results(), subset=subset, task_metadata_og=_legacy_df())
        assert _kept(out) == expected

    def test_predicates_never_touch_df_results(self):
        # df_results carries no metadata columns (max_train_rows/n_features/...); subsetting must
        # still work, proving predicates run on the task grid, not on df_results.
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())
        out = subset_tasks(df_results=_df_results(), subset=["tabpfn"], task_metadata_og=coll)
        # tabpfn: max_train_rows<=10000 & n_features<=500 & n_classes<=10 -> small_ds (binary) and
        # big_ds (regression, n_classes -1 <= 10) is excluded only by size (50k>10k) -> small_ds.
        assert _kept(out) == {("small_ds", 0), ("small_ds", 1)}
