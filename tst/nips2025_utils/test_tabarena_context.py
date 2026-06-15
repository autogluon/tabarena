from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.abstract_arena_context import AbstractArenaContext
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
            "n_samples_test_per_fold": [50, 25_000],
            "NumberOfInstances": [150, 75_000],
            "problem_type": ["binary", "regression"],
            "n_features": [10, 10],
            "n_classes": [2, 0],
        },
    )
    return TabArenaContext(methods=[], task_metadata=TaskMetadataCollection.from_legacy_df(task_metadata))


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
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 0, 0), ("big_ds", 0, 0)]
        assert runner.cache_mode == "only"  # extra kwargs forwarded

    def test_materialize_makes_the_scoped_collection_runnable(self, tmp_path, monkeypatch):
        # materialize=True materializes the (scoped) collection before the runner is built.
        from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection

        calls: list[int] = []
        orig = TaskMetadataCollection.materialize

        def spy(self):
            calls.append(len(self))
            return orig(self)

        monkeypatch.setattr(TaskMetadataCollection, "materialize", spy)
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path), subset="lite", materialize=True)
        # Materialized exactly the scoped collection (2 lite tasks), and the runner still sees them.
        assert calls == [2]
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_no_subset_leaves_full_grid(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path))
        # No filter -> the runner gets the full, unfiltered collection.
        assert runner.task_metadata_collection.dataset_fold_repeats() == [
            ("small_ds", 0, 0),
            ("small_ds", 1, 0),
            ("big_ds", 0, 0),
            ("big_ds", 1, 0),
        ]

    def test_datasets_filter(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(expname=str(tmp_path), datasets=["small_ds"])
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_dataset_fold_repeats_used_as_is_without_subset(self, tmp_path):
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            dataset_fold_repeats=[("big_ds", 1, 0)],
        )
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("big_ds", 1, 0)]

    def test_dataset_fold_repeats_intersected_with_subset(self, tmp_path):
        # "lite" keeps split 0 only -> (small_ds,1,0) is dropped.
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            subset="lite",
            dataset_fold_repeats=[("small_ds", 0, 0), ("small_ds", 1, 0), ("big_ds", 0, 0)],
        )
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_dataset_fold_repeats_intersected_with_datasets(self, tmp_path):
        # datasets=["small_ds"] -> (big_ds,0,0) is dropped.
        runner = _ctx().make_experiment_batch_runner(
            expname=str(tmp_path),
            datasets=["small_ds"],
            dataset_fold_repeats=[("small_ds", 1, 0), ("big_ds", 0, 0)],
        )
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 1, 0)]


def _ctx_multi() -> TabArenaContext:
    # "a": 3 folds x 2 repeats -> splits 0..5; "b": 2 folds x 1 repeat -> splits 0..1.
    task_metadata = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["a", "b"],
            "n_folds": [3, 2],
            "n_repeats": [2, 1],
            "n_samples_train_per_fold": [100, 100],
            "n_samples_test_per_fold": [50, 50],
            "NumberOfInstances": [150, 150],
            "problem_type": ["binary", "binary"],
            "n_features": [10, 10],
            "n_classes": [2, 2],
        },
    )
    return TabArenaContext(methods=[], task_metadata=TaskMetadataCollection.from_legacy_df(task_metadata))


class TestSplitsFoldsRepeats:
    def _dfr(self, tmp_path, **kwargs):
        runner = _ctx_multi().make_experiment_batch_runner(expname=str(tmp_path), **kwargs)
        return runner.task_metadata_collection.dataset_fold_repeats()

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


def _complete_legacy_df() -> pd.DataFrame:
    """A complete legacy frame (all columns `from_legacy_df` requires)."""
    return pd.DataFrame(
        {
            "tid": [363612],
            "dataset": ["ds"],
            "name": ["ds"],
            "problem_type": ["binary"],
            "n_folds": [3],
            "n_repeats": [1],
            "n_features": [7],
            "n_classes": [2],
            "NumberOfInstances": [100],
            "n_samples_train_per_fold": [66],
            "n_samples_test_per_fold": [34],
            "target_feature": ["target"],
        },
    )


class TestNativeTaskMetadata:
    """TabArenaContext takes only the "tabarena" preset or a TaskMetadataCollection.

    DataFrame / list inputs are rejected — the caller must wrap them explicitly so the
    (lossy) legacy conversion is opt-in.
    """

    def test_collection_input_is_held_and_derives_legacy_df(self):
        coll = TaskMetadataCollection.from_legacy_df(_complete_legacy_df())
        ctx = TabArenaContext(methods=[], task_metadata=coll)
        assert ctx.task_metadata_collection is coll
        assert isinstance(ctx.task_metadata, pd.DataFrame)  # derived legacy view
        assert sorted(ctx.task_metadata["dataset"]) == ["ds"]

    def test_list_input_rejected(self):
        # list[TabArenaTaskMetadata] is no longer accepted; wrap in TaskMetadataCollection(...).
        tasks = TaskMetadataCollection.from_legacy_df(_complete_legacy_df()).tasks
        with pytest.raises(TypeError, match="TaskMetadataCollection"):
            TabArenaContext(methods=[], task_metadata=tasks)

    def test_dataframe_input_rejected(self):
        # A legacy DataFrame is no longer accepted; wrap with from_legacy_df.
        with pytest.raises(TypeError, match="from_legacy_df"):
            TabArenaContext(methods=[], task_metadata=_complete_legacy_df())


def _ctx_collection() -> TabArenaContext:
    """Context backed by a native TaskMetadataCollection (2 datasets, 2 folds x 1 repeat)."""
    df = pd.DataFrame(
        {
            "tid": [0, 1],
            "dataset": ["small_ds", "big_ds"],
            "name": ["small_ds", "big_ds"],
            "problem_type": ["binary", "regression"],
            "n_folds": [2, 2],
            "n_repeats": [1, 1],
            "n_features": [10, 10],
            "n_classes": [2, 0],
            "NumberOfInstances": [150, 75_000],
            "n_samples_train_per_fold": [100, 50_000],
            "n_samples_test_per_fold": [50, 25_000],
            "target_feature": ["t", "t"],
        },
    )
    return TabArenaContext(methods=[], task_metadata=TaskMetadataCollection.from_legacy_df(df))


class TestNativeGridSubset:
    """`_subset_dataset_fold_repeats` builds the grid natively from the collection's splits."""

    def test_full_grid_from_collection(self):
        ctx = _ctx_collection()
        assert ctx.task_metadata_collection is not None
        assert set(ctx._subset_dataset_fold_repeats()) == {
            ("small_ds", 0, 0),
            ("small_ds", 1, 0),
            ("big_ds", 0, 0),
            ("big_ds", 1, 0),
        }

    def test_lite_keeps_split_zero(self):
        assert set(_ctx_collection()._subset_dataset_fold_repeats("lite")) == {
            ("small_ds", 0, 0),
            ("big_ds", 0, 0),
        }

    def test_small_predicate_keeps_small_dataset_full_grid(self):
        # "small" == max_train_rows <= 10000 -> only small_ds (n_train=100), both folds.
        assert set(_ctx_collection()._subset_dataset_fold_repeats("small")) == {
            ("small_ds", 0, 0),
            ("small_ds", 1, 0),
        }


class TestAbstractArenaContextStandalone:
    """The base class is directly instantiable with explicit methods/task metadata."""

    @staticmethod
    def _base_ctx() -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_ctx().task_metadata_collection)

    def test_defaults_are_arena_agnostic(self):
        ctx = self._base_ctx()
        assert ctx.methods == []
        assert ctx.load_results().empty  # no methods -> no baseline results
        # Base ships only the universal predicates (no TabArena size/foundation buckets).
        assert set(ctx.subset_predicates) == {"all", "binary", "multiclass", "classification", "regression", "lite"}

    def test_make_experiment_batch_runner_and_subset(self, tmp_path):
        runner = self._base_ctx().make_experiment_batch_runner(expname=str(tmp_path), subset="lite")
        # "lite" keeps split 0 == (fold 0, repeat 0) for each dataset.
        assert runner.task_metadata_collection.dataset_fold_repeats() == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_base_has_no_presets(self):
        with pytest.raises(ValueError, match="defines no presets"):
            AbstractArenaContext(methods="tabarena", task_metadata=_ctx().task_metadata_collection)
        with pytest.raises(ValueError, match="defines no presets"):
            AbstractArenaContext(methods=[], task_metadata="tabarena")


class TestCompareFoldSimilarityForwarding:
    """compare(compute_fold_similarity=, fold_similarity_kwargs=) reaches the lower-level compare."""

    @staticmethod
    def _base_ctx() -> AbstractArenaContext:
        return AbstractArenaContext(methods=[], task_metadata=_ctx().task_metadata_collection)

    @staticmethod
    def _capture(monkeypatch) -> dict:
        import tabarena.nips2025_utils.compare as compare_mod

        captured: dict = {}

        def fake_compare(**kwargs):
            captured.update(kwargs)
            return pd.DataFrame()

        monkeypatch.setattr(compare_mod, "compare", fake_compare)
        return captured

    def test_forwards_flag_and_kwargs(self, monkeypatch, tmp_path):
        captured = self._capture(monkeypatch)
        self._base_ctx().compare(
            output_dir=tmp_path,
            compute_fold_similarity=True,
            fold_similarity_kwargs={"similarity": "pearson", "target_reliability": 0.95},
        )
        assert captured["compute_fold_similarity"] is True
        assert captured["fold_similarity_kwargs"] == {"similarity": "pearson", "target_reliability": 0.95}

    def test_defaults_are_off(self, monkeypatch, tmp_path):
        captured = self._capture(monkeypatch)
        self._base_ctx().compare(output_dir=tmp_path)
        assert captured["compute_fold_similarity"] is False
        assert captured["fold_similarity_kwargs"] is None
