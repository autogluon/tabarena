"""Tests for the Job serialization surface: Job.to_dict/from_dict, build_jobs,
filter_jobs_by_constraints, job_cache_exists, and the JobBatch directory artifact.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.experiment import (
    ExperimentBatchRunner,
    Job,
    JobBatch,
    ModelConstraints,
    build_jobs,
    filter_jobs_by_constraints,
    job_cache_exists,
    task_cache_key_from_task_id_str,
)
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.utils.cache import CacheFunctionPickle

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_experiment(name: str = "lgbm_test", *, hp: dict | None = None):
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment

    return AGModelBagExperiment(
        name=name,
        model_cls=LGBModel,
        model_hyperparameters=hp or {},
        num_bag_folds=2,
        time_limit=60,
    )


def _collection(tids, datasets, *, n_folds=1, n_samples=100, n_features=5) -> TaskMetadataCollection:
    """A small native collection built through the legacy bridge (carries shapes)."""

    def _col(value):
        return value if isinstance(value, list) else [value] * len(tids)

    df = pd.DataFrame(
        {
            "tid": tids,
            "dataset": datasets,
            "name": datasets,
            "problem_type": _col("binary"),
            "n_folds": _col(n_folds),
            "n_repeats": _col(1),
            "n_features": _col(n_features),
            "n_classes": _col(2),
            "NumberOfInstances": _col(200),
            "n_samples_train_per_fold": [float(x) for x in _col(n_samples)],
            "n_samples_test_per_fold": _col(50.0),
        },
    )
    return TaskMetadataCollection.from_legacy_df(df)


# ---------------------------------------------------------------------------
# Job.to_dict / from_dict (inline single-job form)
# ---------------------------------------------------------------------------


class TestJobDictRoundTrip:
    def test_round_trip(self):
        job = Job.create(_make_experiment("exp_a"), "ds_a", fold=1, repeat=2)
        restored = Job.from_dict(job.to_dict())
        assert restored.task == job.task
        assert restored.experiment.name == "exp_a"
        assert restored.experiment.to_yaml_str() == job.experiment.to_yaml_str()

    def test_dict_is_self_contained(self):
        data = Job.create(_make_experiment("exp_a"), "ds_a", fold=0).to_dict()
        assert data["experiment"]["name"] == "exp_a"  # experiment inlined, not referenced
        assert (data["dataset"], data["fold"], data["repeat"]) == ("ds_a", 0, 0)


# ---------------------------------------------------------------------------
# build_jobs — the shared grid enumerator
# ---------------------------------------------------------------------------


class TestBuildJobs:
    def test_expands_experiments_x_splits_in_task_split_experiment_order(self):
        collection = _collection([1, 2], ["ds_a", "ds_b"], n_folds=[2, 1])
        exp_a, exp_b = _make_experiment("exp_a"), _make_experiment("exp_b")
        jobs = build_jobs([exp_a, exp_b], collection)
        assert [(j.experiment.name, *j.task.as_triple()) for j in jobs] == [
            ("exp_a", "ds_a", 0, 0),
            ("exp_b", "ds_a", 0, 0),
            ("exp_a", "ds_a", 1, 0),
            ("exp_b", "ds_a", 1, 0),
            ("exp_a", "ds_b", 0, 0),
            ("exp_b", "ds_b", 0, 0),
        ]

    def test_sparse_collection_respected(self):
        collection = _collection([1, 2], ["ds_a", "ds_b"], n_folds=3).subset(
            [("ds_a", 0, 0), ("ds_b", 2, 0)],
        )
        jobs = build_jobs([_make_experiment()], collection)
        assert [j.task.as_triple() for j in jobs] == [("ds_a", 0, 0), ("ds_b", 2, 0)]


# ---------------------------------------------------------------------------
# filter_jobs_by_constraints
# ---------------------------------------------------------------------------


class TestFilterJobsByConstraints:
    def test_filters_by_shape_per_split(self):
        collection = _collection([1, 2], ["small", "big"], n_samples=[100, 100_000])
        jobs = build_jobs([_make_experiment()], collection)
        # LGBModel's AG key is "GBM": constrain it to <= 10_000 train samples.
        kept = filter_jobs_by_constraints(
            jobs,
            model_constraints={"GBM": ModelConstraints(max_n_samples_train_per_fold=10_000)},
            task_metadata=collection,
        )
        assert [j.task.dataset for j in kept] == ["small"]

    def test_unconstrained_model_passes(self):
        collection = _collection([1], ["big"], n_samples=100_000)
        jobs = build_jobs([_make_experiment()], collection)
        kept = filter_jobs_by_constraints(
            jobs,
            model_constraints={"SOME_OTHER_KEY": ModelConstraints(max_n_samples_train_per_fold=1)},
            task_metadata=collection,
        )
        assert kept == jobs

    def test_unknown_split_raises(self):
        collection = _collection([1], ["ds_a"])
        job = Job.create(_make_experiment(), "ds_a", fold=7)
        with pytest.raises(ValueError, match="not a split"):
            filter_jobs_by_constraints(
                [job],
                model_constraints={},
                task_metadata=collection,
            )

    def test_run_jobs_opt_in_filters_everything_without_running(self, tmp_path):
        """When constraints filter out every job, run_jobs returns [] without any task load."""
        collection = _collection([1], ["big"], n_samples=100_000)
        runner = ExperimentBatchRunner(expname=str(tmp_path), task_metadata=collection)
        results = runner.run_jobs(
            build_jobs([_make_experiment()], collection),
            model_constraints={"GBM": ModelConstraints(max_n_samples_train_per_fold=10)},
        )
        assert results == []


# ---------------------------------------------------------------------------
# job_cache_exists — writer-aligned cache-hit check
# ---------------------------------------------------------------------------


class TestJobCacheExists:
    def test_task_cache_key_normalization(self):
        assert task_cache_key_from_task_id_str("363612") == 363612
        user_key = task_cache_key_from_task_id_str("UserTask|9900335484|ds/uuid")
        assert isinstance(user_key, str)  # UserTask slug, not an int

    def test_miss_then_hit(self, tmp_path):
        kwargs = {
            "output_dir": str(tmp_path),
            "method_name": "exp_a",
            "task_id_str": "42",
            "fold": 1,
            "repeat": 0,
        }
        assert not job_cache_exists(**kwargs)
        # Write through the same layout the engine writes: data/{method}/{task}/{repeat}_{fold}.
        cacher = CacheFunctionPickle(cache_name="results", cache_path=str(tmp_path / "data" / "exp_a" / "42" / "0_1"))
        cacher.save_cache({"metric_error": 0.1})
        assert job_cache_exists(**kwargs)
        # A different coordinate still misses.
        assert not job_cache_exists(**{**kwargs, "fold": 0})


# ---------------------------------------------------------------------------
# JobBatch — validation + directory round trip
# ---------------------------------------------------------------------------


class TestJobBatch:
    def _batch(self) -> JobBatch:
        collection = _collection([1, 2], ["ds_a", "ds_b"], n_folds=[2, 1])
        exp_a, exp_b = _make_experiment("exp_a"), _make_experiment("exp_b", hp={"learning_rate": 0.05})
        # Non-rectangular on purpose: exp_b only runs on ds_a fold 0.
        jobs = [
            Job.create(exp_a, "ds_a", fold=0),
            Job.create(exp_a, "ds_a", fold=1),
            Job.create(exp_a, "ds_b", fold=0),
            Job.create(exp_b, "ds_a", fold=0),
        ]
        return JobBatch(jobs=jobs, task_metadata=collection)

    def test_name_collision_between_different_experiments_raises(self):
        collection = _collection([1], ["ds_a"])
        jobs = [
            Job.create(_make_experiment("same"), "ds_a", fold=0),
            Job.create(_make_experiment("same", hp={"learning_rate": 0.05}), "ds_a", fold=0),
        ]
        with pytest.raises(ValueError, match="share the name"):
            JobBatch(jobs=jobs, task_metadata=collection)

    def test_job_split_not_in_collection_raises(self):
        collection = _collection([1], ["ds_a"], n_folds=1)
        with pytest.raises(ValueError, match="not in `task_metadata`"):
            JobBatch(jobs=[Job.create(_make_experiment(), "ds_a", fold=5)], task_metadata=collection)

    def test_experiments_deduplicated_in_first_seen_order(self):
        batch = self._batch()
        assert [e.name for e in batch.experiments] == ["exp_a", "exp_b"]

    def test_save_load_round_trip(self, tmp_path):
        batch = self._batch()
        batch.save(tmp_path / "batch")
        loaded = JobBatch.load(tmp_path / "batch")

        assert [(j.experiment.name, *j.task.as_triple()) for j in loaded.jobs] == [
            (j.experiment.name, *j.task.as_triple()) for j in batch.jobs
        ]
        # Compare the parsed dict form (YAML key order is not part of the identity).
        assert [e.to_yaml_dict() for e in loaded.experiments] == [e.to_yaml_dict() for e in batch.experiments]
        assert loaded.task_metadata.dataset_fold_repeats() == batch.task_metadata.dataset_fold_repeats()
        # Shared experiments stay shared after load (stored once, referenced by name).
        exp_a_jobs = [j for j in loaded.jobs if j.experiment.name == "exp_a"]
        assert all(j.experiment is exp_a_jobs[0].experiment for j in exp_a_jobs)

    def test_load_with_unknown_experiment_reference_raises(self, tmp_path):
        batch = self._batch()
        path = batch.save(tmp_path / "batch")
        jobs_file = path / "jobs.json"
        jobs_file.write_text(jobs_file.read_text().replace("exp_b", "exp_missing"))
        with pytest.raises(ValueError, match="exp_missing"):
            JobBatch.load(path)
