from __future__ import annotations

import pandas as pd

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
