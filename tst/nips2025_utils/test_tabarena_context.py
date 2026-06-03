from __future__ import annotations

import pandas as pd
import pytest

from tabarena.nips2025_utils.tabarena_context import TabArenaContext


def _ctx() -> TabArenaContext:
    # methods=[] keeps construction light (no method-metadata collection load).
    task_metadata = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["small_ds", "big_ds"],
            "n_folds": [2, 2],
            "n_repeats": [1, 1],
            "n_samples_train_per_fold": [100, 50_000],
            "problem_type": ["binary", "regression"],
            "n_features": [10, 10],
            "n_classes": [2, 0],
        },
    )
    return TabArenaContext(methods=[], task_metadata=task_metadata)


class TestSubsetDatasetFoldRepeats:
    """subset -> (dataset, fold, repeat) triplets, where split = n_folds*repeat + fold."""

    def test_lite_keeps_split_zero_per_dataset(self):
        # "lite" == split 0 == (fold 0, repeat 0) for every dataset.
        assert _ctx()._subset_dataset_fold_repeats("lite") == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_dataset_predicate_keeps_full_grid_of_matching_datasets(self):
        # "small" == max_train_rows <= 10000 -> only small_ds, all (fold, repeat).
        assert _ctx()._subset_dataset_fold_repeats("small") == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_predicates_are_anded(self):
        assert _ctx()._subset_dataset_fold_repeats(["small", "lite"]) == [("small_ds", 0, 0)]


class TestMakeExperimentBatchRunner:
    def test_subset_forwarded_as_dataset_fold_repeats(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path), subset="lite", cache_mode="only")
        assert runner._dataset_fold_repeats == [("small_ds", 0, 0), ("big_ds", 0, 0)]
        assert runner.cache_mode == "only"  # extra kwargs forwarded

    def test_no_subset_leaves_full_grid(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path))
        assert runner._dataset_fold_repeats is None

    def test_datasets_filter(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path), datasets=["small_ds"])
        assert runner._dataset_fold_repeats == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_dataset_fold_repeats_used_as_is_without_subset(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            dataset_fold_repeats=[("big_ds", 1, 0)],
        )
        assert runner._dataset_fold_repeats == [("big_ds", 1, 0)]

    def test_dataset_fold_repeats_intersected_with_subset(self, tmp_path):
        # "lite" keeps split 0 only -> (small_ds,1,0) is dropped.
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            subset="lite",
            dataset_fold_repeats=[("small_ds", 0, 0), ("small_ds", 1, 0), ("big_ds", 0, 0)],
        )
        assert runner._dataset_fold_repeats == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_dataset_fold_repeats_intersected_with_datasets(self, tmp_path):
        # datasets=["small_ds"] -> (big_ds,0,0) is dropped.
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            datasets=["small_ds"],
            dataset_fold_repeats=[("small_ds", 1, 0), ("big_ds", 0, 0)],
        )
        assert runner._dataset_fold_repeats == [("small_ds", 1, 0)]


def _ctx_multi() -> TabArenaContext:
    # "a": 3 folds x 2 repeats -> splits 0..5; "b": 2 folds x 1 repeat -> splits 0..1.
    task_metadata = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["a", "b"],
            "n_folds": [3, 2],
            "n_repeats": [2, 1],
            "n_samples_train_per_fold": [100, 100],
            "problem_type": ["binary", "binary"],
            "n_features": [10, 10],
            "n_classes": [2, 2],
        },
    )
    return TabArenaContext(methods=[], task_metadata=task_metadata)


class TestSplitsFoldsRepeats:
    def _dfr(self, tmp_path, **kwargs):
        return _ctx_multi().make_experiment_batch_runner(expname=str(tmp_path), **kwargs)._dataset_fold_repeats

    def test_folds_filter(self, tmp_path):
        # fold 0 across all repeats: a(0,0),(0,1); b(0,0).
        assert self._dfr(tmp_path, folds=[0]) == [("a", 0, 0), ("a", 0, 1), ("b", 0, 0)]

    def test_repeats_filter(self, tmp_path):
        # repeat 1 only exists for "a" (b has a single repeat).
        assert self._dfr(tmp_path, repeats=[1]) == [("a", 0, 1), ("a", 1, 1), ("a", 2, 1)]

    def test_folds_and_repeats_compose(self, tmp_path):
        assert self._dfr(tmp_path, folds=[0], repeats=[0]) == [("a", 0, 0), ("b", 0, 0)]

    def test_splits_filter(self, tmp_path):
        # split 3 for "a" (n_folds=3) == (fold 0, repeat 1); "b" has no split 3.
        assert self._dfr(tmp_path, splits=[3]) == [("a", 0, 1)]

    def test_splits_compose_with_datasets(self, tmp_path):
        assert self._dfr(tmp_path, datasets=["a"], splits=[0]) == [("a", 0, 0)]

    def test_splits_with_folds_raises(self, tmp_path):
        with pytest.raises(ValueError, match="`splits` together with `folds`"):
            self._dfr(tmp_path, splits=[0], folds=[0])

    def test_splits_with_repeats_raises(self, tmp_path):
        with pytest.raises(ValueError, match="`splits` together with `folds`"):
            self._dfr(tmp_path, splits=[0], repeats=[0])

    def test_dataset_fold_repeats_with_folds_raises(self, tmp_path):
        with pytest.raises(ValueError, match="`dataset_fold_repeats` together"):
            self._dfr(tmp_path, dataset_fold_repeats=[("a", 0, 0)], folds=[0])

    def test_dataset_fold_repeats_with_splits_raises(self, tmp_path):
        with pytest.raises(ValueError, match="`dataset_fold_repeats` together"):
            self._dfr(tmp_path, dataset_fold_repeats=[("a", 0, 0)], splits=[0])
