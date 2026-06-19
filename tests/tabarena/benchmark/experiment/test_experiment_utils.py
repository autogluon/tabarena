from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.experiment.experiment_runner_api import (
    _build_cache_prefix,
    _iter_jobs,
    _JobSpec,
    _RunStats,
)
from tabarena.utils.cache import AbstractCacheFunction


class _RecordingCache(AbstractCacheFunction):
    """Minimal cache class that records its constructions and never has a cache hit."""

    instances: list = []

    def __init__(self, cache_name, cache_path, **kwargs):
        super().__init__(include_self_in_call=kwargs.get("include_self_in_call", False))
        self.cache_name = cache_name
        self.cache_path = cache_path
        type(self).instances.append((cache_name, cache_path, kwargs))

    @property
    def cache_file(self):
        return None

    @property
    def exists(self) -> bool:
        return False

    def save_cache(self, data) -> None:
        pass

    def load_cache(self):
        return None


class _IdentityHitCache(AbstractCacheFunction):
    """Cache that always 'hits' and loads back an identifier derived from its path.

    Lets a `cache_mode="only"` run return one result per job without any training, where
    each result encodes which (method, tid, repeat_fold) it came from — so result ordering
    can be asserted against the input.
    """

    def __init__(self, cache_name, cache_path, **kwargs):
        super().__init__(include_self_in_call=kwargs.get("include_self_in_call", False))
        self.cache_name = cache_name
        self.cache_path = cache_path

    @property
    def cache_file(self):
        return None

    @property
    def exists(self) -> bool:
        return True

    def save_cache(self, data) -> None:
        pass

    def load_cache(self):
        return {"suffix": self.cache_path.rsplit("/data/", 1)[1]}


# ---------------------------------------------------------------------------
# ExperimentBatchRunner._run_individual — input validation (no ML training)
# ---------------------------------------------------------------------------


def _make_minimal_experiment(name: str = "lgbm_test"):
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment

    return AGModelBagExperiment(
        name=name,
        model_cls=LGBModel,
        model_hyperparameters={},
        num_bag_folds=2,
        time_limit=60,
    )


def _make_experiment_variant(name: str, *, hp: dict):
    """An experiment with a given name and hyperparameters (to build name collisions)."""
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment

    return AGModelBagExperiment(
        name=name,
        model_cls=LGBModel,
        model_hyperparameters=hp,
        num_bag_folds=2,
        time_limit=60,
    )


def _legacy_collection(tids, datasets, *, n_folds=1, n_repeats=1, problem_type="binary"):
    """Build a TaskMetadataCollection from a compact (tid, dataset) spec for the runner tests.

    ExperimentBatchRunner only accepts a TaskMetadataCollection, so this fills the columns
    `from_legacy_df` requires. `n_folds`/`n_repeats` accept an int (shared) or a per-dataset
    list (controls the collection's split grid, which `run_all` / init validation key on).
    """
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    def _col(value):
        return value if isinstance(value, list) else [value] * len(tids)

    df = pd.DataFrame(
        {
            "tid": tids,
            "dataset": datasets,
            "name": datasets,
            "problem_type": _col(problem_type),
            "n_folds": _col(n_folds),
            "n_repeats": _col(n_repeats),
            "n_features": _col(5),
            "n_classes": _col(2),
            "NumberOfInstances": _col(100),
            "n_samples_train_per_fold": _col(66),
            "n_samples_test_per_fold": _col(34),
            "target_feature": _col("t"),
        },
    )
    return TaskMetadataCollection.from_legacy_df(df)


class TestRunIndividualValidation:
    @staticmethod
    def _runner(tmp_path):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        return ExperimentBatchRunner(
            expname=str(tmp_path),
            task_metadata=_legacy_collection([360], ["d0"]),
            cache_mode="only",
        )

    def test_non_experiment_in_methods_raises(self, tmp_path):
        with pytest.raises(AssertionError, match="instances of Experiment"):
            self._runner(tmp_path).run(methods=["not_an_experiment"], datasets=["d0"], folds=[0])

    def test_duplicate_experiment_names_raises(self, tmp_path):
        exp1 = _make_minimal_experiment("dup_name")
        exp2 = _make_minimal_experiment("dup_name")
        with pytest.raises(AssertionError, match="Duplicate experiment name"):
            self._runner(tmp_path).run(methods=[exp1, exp2], datasets=["d0"], folds=[0])


class TestBuildCachePrefix:
    def test_name_first_with_repeat(self):
        prefix = _build_cache_prefix(
            method_name="m",
            cache_task_key=360,
            fold=2,
            repeat=1,
        )
        assert prefix == "data/m/360/1_2"


class TestExperimentBatchRunnerDelegation:
    """ExperimentBatchRunner.run() builds job specs straight into the shared engine."""

    @staticmethod
    def _runner(tmp_path, **kwargs):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        return ExperimentBatchRunner(
            expname=str(tmp_path),
            task_metadata=_legacy_collection([0, 1], ["d0", "d1"]),
            **kwargs,
        )

    def test_only_cache_no_cache_returns_empty(self, tmp_path):
        # cache_mode="only"; with no cache present, returns [] without
        # loading any task or training (exercises the full delegation wiring).
        runner = self._runner(tmp_path, cache_mode="only")
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0", "d1"],
            folds=[0],
        )
        assert result == []

    def test_only_cache_with_repeats(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0"],
            folds=[0, 1],
            repeats=[0, 1],
        )
        assert result == []

    def test_custom_cache_cls_is_forwarded(self, tmp_path):
        # A non-Pickle cache class is threaded through to the engine and used.
        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_RecordingCache)
        result = runner.run(
            methods=[_make_minimal_experiment()],
            datasets=["d0"],
            folds=[0],
        )
        assert result == []
        assert _RecordingCache.instances, "custom cache_cls was not used"
        # ExperimentBatchRunner's default cache_cls_kwargs forwards include_self_in_call=True.
        assert all(kw.get("include_self_in_call") is True for *_, kw in _RecordingCache.instances)

    def test_unknown_dataset_raises(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        with pytest.raises(ValueError, match="present in task_metadata"):
            runner.run(
                methods=[_make_minimal_experiment()],
                datasets=["does_not_exist"],
                folds=[0],
            )


class TestExperimentBatchRunnerRunAll:
    """ExperimentBatchRunner.run_dataset_fold_repeats and run_all."""

    @staticmethod
    def _runner(tmp_path, task_metadata=None, **kwargs):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        if task_metadata is None:
            task_metadata = _legacy_collection([0, 1], ["d0", "d1"])
        return ExperimentBatchRunner(expname=str(tmp_path), task_metadata=task_metadata, **kwargs)

    @staticmethod
    def _cache_suffixes() -> list[str]:
        # cache_path is `{expname}/data/{method}/{tid}/{repeat}_{fold}`.
        return sorted(cache_path.rsplit("/data/", 1)[1] for _, cache_path, _ in _RecordingCache.instances)

    def test_run_dataset_fold_repeats_runs_exact_triples(self, tmp_path):
        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_RecordingCache)
        result = runner.run_dataset_fold_repeats(
            methods=[_make_minimal_experiment("m")],
            dataset_fold_repeats=[("d0", 0, 0), ("d0", 1, 0), ("d1", 2, 1)],
        )
        assert result == []
        # Exactly the requested (tid, repeat_fold) cache lookups — no cartesian product.
        assert self._cache_suffixes() == ["m/0/0_0", "m/0/0_1", "m/1/1_2"]

    def test_run_dataset_fold_repeats_duplicate_triples_raises(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        with pytest.raises(AssertionError, match="Duplicate"):
            runner.run_dataset_fold_repeats(
                methods=[_make_minimal_experiment()],
                dataset_fold_repeats=[("d0", 0, 0), ("d0", 0, 0)],
            )

    def test_run_dataset_fold_repeats_unknown_dataset_raises(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        with pytest.raises(ValueError, match="present in task_metadata"):
            runner.run_dataset_fold_repeats(
                methods=[_make_minimal_experiment()],
                dataset_fold_repeats=[("does_not_exist", 0, 0)],
            )

    def test_run_all_expands_metadata_folds_and_repeats(self, tmp_path):
        _RecordingCache.instances.clear()
        task_metadata = _legacy_collection([0, 1], ["d0", "d1"], n_folds=[2, 1], n_repeats=[1, 2])
        runner = self._runner(tmp_path, task_metadata=task_metadata, cache_mode="only", cache_cls=_RecordingCache)
        result = runner.run_all(methods=[_make_minimal_experiment("m")])
        assert result == []
        # d0: 2 folds x 1 repeat -> (f0,r0),(f1,r0); d1: 1 fold x 2 repeats -> (f0,r0),(f0,r1)
        assert self._cache_suffixes() == ["m/0/0_0", "m/0/0_1", "m/1/0_0", "m/1/1_0"]

    def test_legacy_dataframe_rejected(self, tmp_path):
        # Legacy DataFrame input is no longer accepted; callers must wrap with from_legacy_df.
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        with pytest.raises(TypeError, match="from_legacy_df"):
            ExperimentBatchRunner(
                expname=str(tmp_path),
                task_metadata=pd.DataFrame({"tid": [0], "dataset": ["d0"]}),
            )

    def test_only_strict_raises_on_missing_cache(self, tmp_path):
        # cache_mode="only_strict": no cache exists, so every requested experiment is
        # missing and the run raises (rather than silently returning []).
        runner = self._runner(tmp_path, cache_mode="only_strict")
        with pytest.raises(AssertionError, match="only_strict"):
            runner.run(methods=[_make_minimal_experiment()], datasets=["d0", "d1"], folds=[0])

    def test_only_strict_generalizes_to_run_all(self, tmp_path):
        # only_strict works through run_all too (same canonical cache path).
        runner = self._runner(tmp_path, cache_mode="only_strict")
        with pytest.raises(AssertionError, match="only_strict"):
            runner.run_all(methods=[_make_minimal_experiment()])

    def test_run_all_runs_pre_filtered_collection(self, tmp_path):
        # Scoping a run_all to a subset is done by pre-filtering the collection (no
        # dataset_fold_repeats arg): run_all runs exactly the collection's splits.
        _RecordingCache.instances.clear()
        # d1 has 3 folds x 2 repeats, so (d1, fold 2, repeat 1) is a real split.
        collection = _legacy_collection([0, 1], ["d0", "d1"], n_folds=[1, 3], n_repeats=[1, 2]).subset(
            [("d0", 0, 0), ("d1", 2, 1)],
        )
        runner = self._runner(tmp_path, task_metadata=collection, cache_mode="only", cache_cls=_RecordingCache)
        result = runner.run_all(methods=[_make_minimal_experiment("m")])
        assert result == []
        assert self._cache_suffixes() == ["m/0/0_0", "m/1/1_2"]

    def test_subset_invalid_split_raises(self, tmp_path):
        # repeat=5 is not a real split for d0 (2 folds x 1 repeat) -> subset rejects it.
        collection = _legacy_collection([0], ["d0"], n_folds=2, n_repeats=1)
        with pytest.raises(ValueError, match="not splits of this collection"):
            collection.subset([("d0", 0, 5)])


class TestExperimentBatchRunnerNativeCollection:
    """ExperimentBatchRunner accepts a TaskMetadataCollection and derives tid map / grid natively."""

    @staticmethod
    def _collection():
        from tabarena.benchmark.task.metadata import TaskMetadataCollection

        df = pd.DataFrame(
            {
                "tid": [10, 11],
                "dataset": ["d0", "d1"],
                "name": ["d0", "d1"],
                "problem_type": ["binary", "binary"],
                "n_folds": [2, 1],
                "n_repeats": [1, 2],
                "n_features": [5, 5],
                "n_classes": [2, 2],
                "NumberOfInstances": [100, 100],
                "n_samples_train_per_fold": [66, 66],
                "n_samples_test_per_fold": [34, 34],
                "target_feature": ["t", "t"],
            },
        )
        return TaskMetadataCollection.from_legacy_df(df)

    def _runner(self, tmp_path, **kwargs):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        return ExperimentBatchRunner(expname=str(tmp_path), task_metadata=self._collection(), **kwargs)

    @staticmethod
    def _cache_suffixes() -> list[str]:
        return sorted(cache_path.rsplit("/data/", 1)[1] for _, cache_path, _ in _RecordingCache.instances)

    def test_tid_map_from_collection(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        assert runner.task_metadata_collection is not None
        assert runner._dataset_to_tid_dict == {"d0": 10, "d1": 11}
        assert sorted(runner.datasets) == ["d0", "d1"]

    def test_run_all_expands_from_collection_splits(self, tmp_path):
        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_RecordingCache)
        result = runner.run_all(methods=[_make_minimal_experiment("m")])
        assert result == []
        # d0 (tid 10): 2 folds x 1 repeat; d1 (tid 11): 1 fold x 2 repeats.
        assert self._cache_suffixes() == ["m/10/0_0", "m/10/0_1", "m/11/0_0", "m/11/1_0"]

    def test_invalid_split_rejected_against_actual_splits(self, tmp_path):
        # d1 has 1 fold x 2 repeats, so (d1, fold 1, repeat 0) is not a real split.
        with pytest.raises(ValueError, match="not splits of this collection"):
            self._collection().subset([("d1", 1, 0)])

    def test_unknown_dataset_rejected(self, tmp_path):
        with pytest.raises(ValueError, match="not splits of this collection"):
            self._collection().subset([("nope", 0, 0)])


class TestIterJobsGrouping:
    """_iter_jobs groups specs by task (one _LazyTask per task) while threading input order."""

    def test_groups_by_task_and_threads_input_index(self, tmp_path):
        from types import SimpleNamespace

        from tabarena.benchmark.task.openml import OpenMLTaskSpec

        a = SimpleNamespace(name="a")
        b = SimpleNamespace(name="b")
        # Task 0 appears at indices 0 and 2 (different experiments); task 1 at index 1.
        task0, task1 = OpenMLTaskSpec(0), OpenMLTaskSpec(1)
        specs = [
            _JobSpec(model_experiment=a, task=task0, fold=0, repeat=0),
            _JobSpec(model_experiment=a, task=task1, fold=0, repeat=0),
            _JobSpec(model_experiment=b, task=task0, fold=1, repeat=0),
        ]
        stats = _RunStats(total=len(specs))
        jobs = list(
            _iter_jobs(
                specs,
                stats=stats,
                base_cache_path=str(tmp_path),
                cache_cls=_RecordingCache,
                cache_cls_kwargs=None,
            ),
        )

        # Run order: grouped by task, first-seen task order, input order within a task.
        assert [j.input_index for j in jobs] == [0, 2, 1]
        # One _LazyTask per distinct task, shared across that task's jobs regardless of which
        # experiment they belong to — the load-sharing / one-dataset-resident invariant.
        assert jobs[0].lazy_task is jobs[1].lazy_task  # both task 0
        assert jobs[0].lazy_task is not jobs[2].lazy_task  # task 1 is distinct
        assert len({id(j.lazy_task) for j in jobs}) == 2
        assert stats.started == 3


class TestExperimentBatchRunnerRunJobs:
    """ExperimentBatchRunner.run_jobs: sparse (experiment, task) jobs via the shared engine."""

    @staticmethod
    def _runner(tmp_path, **kwargs):
        from tabarena.benchmark.experiment import ExperimentBatchRunner

        return ExperimentBatchRunner(
            expname=str(tmp_path),
            task_metadata=_legacy_collection([0, 1], ["d0", "d1"]),
            **kwargs,
        )

    @staticmethod
    def _cache_suffixes() -> list[str]:
        return sorted(cache_path.rsplit("/data/", 1)[1] for _, cache_path, _ in _RecordingCache.instances)

    def test_empty_jobs_returns_empty(self, tmp_path):
        runner = self._runner(tmp_path, cache_mode="only")
        assert runner.run_jobs([]) == []

    def test_enumerates_exact_sparse_units(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_RecordingCache)
        a = _make_minimal_experiment("a")
        b = _make_minimal_experiment("b")
        # Non-rectangular: a and b run on different tasks/splits, no cartesian product.
        result = runner.run_jobs(
            [
                Job.create(a, "d0", 0, 0),
                Job.create(b, "d1", 2, 1),
                Job.create(a, "d1", 0, 0),
            ],
        )
        assert result == []
        assert self._cache_suffixes() == ["a/0/0_0", "a/1/0_0", "b/1/1_2"]

    def test_results_match_input_job_order(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_IdentityHitCache)
        a = _make_minimal_experiment("a")
        b = _make_minimal_experiment("b")
        # Deliberately interleave tasks so input order differs from task-grouped run order.
        jobs = [
            Job.create(a, "d1", 0, 0),  # a/1/0_0
            Job.create(b, "d0", 0, 0),  # b/0/0_0
            Job.create(a, "d0", 1, 0),  # a/0/0_1
            Job.create(b, "d1", 2, 1),  # b/1/1_2
        ]
        result = runner.run_jobs(jobs)
        # Results come back in input-job order, not the (task-grouped) execution order.
        assert [r["suffix"] for r in result] == ["a/1/0_0", "b/0/0_0", "a/0/0_1", "b/1/1_2"]

    def test_shared_task_enumerated_once_per_method_only(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        _RecordingCache.instances.clear()
        runner = self._runner(tmp_path, cache_mode="only", cache_cls=_RecordingCache)
        a = _make_minimal_experiment("a")
        b = _make_minimal_experiment("b")
        # Both methods on the same task/split: two distinct cache units (one per method),
        # never duplicated by the per-experiment grouping the old implementation used.
        result = runner.run_jobs([Job.create(a, "d0", 0, 0), Job.create(b, "d0", 0, 0)])
        assert result == []
        assert self._cache_suffixes() == ["a/0/0_0", "b/0/0_0"]

    def test_duplicate_job_raises(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        runner = self._runner(tmp_path, cache_mode="only")
        a = _make_minimal_experiment("a")
        with pytest.raises(AssertionError, match="Duplicate job"):
            runner.run_jobs([Job.create(a, "d0", 0, 0), Job.create(a, "d0", 0, 0)])

    def test_same_name_different_config_raises(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        runner = self._runner(tmp_path, cache_mode="only")
        dup1 = _make_experiment_variant("dup", hp={})
        dup2 = _make_experiment_variant("dup", hp={"learning_rate": 0.05})
        with pytest.raises(AssertionError, match="share the name"):
            runner.run_jobs([Job.create(dup1, "d0", 0, 0), Job.create(dup2, "d1", 0, 0)])

    def test_unknown_dataset_raises(self, tmp_path):
        from tabarena.benchmark.experiment import Job

        runner = self._runner(tmp_path, cache_mode="only")
        a = _make_minimal_experiment("a")
        with pytest.raises(ValueError, match="present in task_metadata"):
            runner.run_jobs([Job.create(a, "nope", 0, 0)])
