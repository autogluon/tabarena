from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection, TaskSubset
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


class _StubExperiment:
    """Minimal stand-in for an Experiment.

    ``build_jobs`` only reads ``.name`` (cache identity) and ``.model_constraints``
    (None -> the job runs on every split), so a stub avoids the AutoGluon dependency.
    """

    model_constraints = None

    def __init__(self, name: str = "exp"):
        self.name = name


class TestBuildJobs:
    """build_jobs scopes the collection via subset_tasks, then enumerates experiment x split jobs."""

    def test_lite_keeps_split_zero_per_dataset(self):
        # "lite" == split 0 == (fold 0, repeat 0) for every dataset.
        jobs = _ctx().build_jobs([_StubExperiment()], subset="lite")
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_subset_predicate_keeps_full_grid_of_matching_datasets(self):
        # "small" == max_train_rows <= 10000 -> only small_ds, all (fold, repeat).
        jobs = _ctx().build_jobs([_StubExperiment()], subset="small")
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_subset_predicates_are_anded(self):
        jobs = _ctx().build_jobs([_StubExperiment()], subset=["small", "lite"])
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0)]

    def test_dataset_names_filter(self):
        # `dataset_names` is forwarded to subset_tasks (the old `datasets` filter).
        jobs = _ctx().build_jobs([_StubExperiment()], dataset_names=["small_ds"])
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_no_filter_uses_full_grid(self):
        jobs = _ctx().build_jobs([_StubExperiment()])
        assert [j.task.as_triple() for j in jobs] == [
            ("small_ds", 0, 0),
            ("small_ds", 1, 0),
            ("big_ds", 0, 0),
            ("big_ds", 1, 0),
        ]

    def test_experiments_crossed_with_splits_in_task_split_experiment_order(self):
        jobs = _ctx().build_jobs([_StubExperiment("a"), _StubExperiment("b")], subset="lite")
        assert [(j.experiment.name, *j.task.as_triple()) for j in jobs] == [
            ("a", "small_ds", 0, 0),
            ("b", "small_ds", 0, 0),
            ("a", "big_ds", 0, 0),
            ("b", "big_ds", 0, 0),
        ]

    def test_task_subset_object_scopes_like_subset_kwarg(self):
        # A typed TaskSubset is equivalent to the loose `subset=` keyword.
        jobs = _ctx().build_jobs([_StubExperiment()], task_subset=TaskSubset(subset="lite"))
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("big_ds", 0, 0)]

    def test_task_subset_dict_resolves(self):
        jobs = _ctx().build_jobs([_StubExperiment()], task_subset={"dataset_names": ["small_ds"]})
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("small_ds", 1, 0)]

    def test_task_subset_object_and_loose_kwarg_combine_per_field(self):
        # Different fields combine: the spec scopes datasets, the loose `subset` adds the split filter.
        jobs = _ctx().build_jobs(
            [_StubExperiment()],
            task_subset=TaskSubset(dataset_names=["small_ds"]),
            subset="lite",
        )
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0)]

    def test_loose_kwarg_overrides_task_subset_same_field(self):
        # Same field: the loose `subset="lite"` overrides the spec's `subset="small"`.
        jobs = _ctx().build_jobs([_StubExperiment()], task_subset=TaskSubset(subset="small"), subset="lite")
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("big_ds", 0, 0)]


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


class TestNativeGridBuildJobs:
    """build_jobs enumerates the grid natively from a native collection's splits."""

    def test_full_grid_from_collection(self):
        jobs = _ctx_collection().build_jobs([_StubExperiment()])
        assert {j.task.as_triple() for j in jobs} == {
            ("small_ds", 0, 0),
            ("small_ds", 1, 0),
            ("big_ds", 0, 0),
            ("big_ds", 1, 0),
        }

    def test_lite_keeps_split_zero(self):
        jobs = _ctx_collection().build_jobs([_StubExperiment()], subset="lite")
        assert {j.task.as_triple() for j in jobs} == {("small_ds", 0, 0), ("big_ds", 0, 0)}

    def test_small_predicate_keeps_small_dataset_full_grid(self):
        # "small" == max_train_rows <= 10000 -> only small_ds (n_train=100), both folds.
        jobs = _ctx_collection().build_jobs([_StubExperiment()], subset="small")
        assert {j.task.as_triple() for j in jobs} == {("small_ds", 0, 0), ("small_ds", 1, 0)}


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

    def test_build_jobs_scopes_via_subset(self):
        jobs = self._base_ctx().build_jobs([_StubExperiment()], subset="lite")
        # "lite" keeps split 0 == (fold 0, repeat 0) for each dataset.
        assert [j.task.as_triple() for j in jobs] == [("small_ds", 0, 0), ("big_ds", 0, 0)]

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
