from __future__ import annotations

import argparse
import os

import pytest

# Import a real submodule (not the bare `tabflow_slurm` namespace): when the package
# is not installed, the repo-root workspace dir is importable as an empty namespace
# package, so `importorskip("tabflow_slurm")` would NOT skip. A submodule does.
pytest.importorskip("tabflow_slurm.run_tabarena_experiment", reason="tabflow_slurm is not installed")

from tabflow_slurm.run_tabarena_experiment import (
    _parse_int_or_none,
    _str2bool,
    _strip_quotes,
    run_experiment,
)
from tabflow_slurm.slurm_utils import setup_slurm_job

# ---------------------------------------------------------------------------
# _str2bool
# ---------------------------------------------------------------------------


class TestStr2Bool:
    @pytest.mark.parametrize("value", ["yes", "true", "t", "1", "YES", "True", "T"])
    def test_truthy_strings(self, value):
        assert _str2bool(value) is True

    @pytest.mark.parametrize("value", ["no", "false", "f", "0", "NO", "False", "F"])
    def test_falsy_strings(self, value):
        assert _str2bool(value) is False

    def test_bool_true_passthrough(self):
        assert _str2bool(True) is True

    def test_bool_false_passthrough(self):
        assert _str2bool(False) is False

    @pytest.mark.parametrize("value", ["maybe", "yes_no", "2", "tru", "fals", ""])
    def test_invalid_raises(self, value):
        with pytest.raises(argparse.ArgumentTypeError):
            _str2bool(value)

    def test_invalid_message_mentions_boolean(self):
        with pytest.raises(argparse.ArgumentTypeError, match="[Bb]oolean"):
            _str2bool("not_a_bool")


# ---------------------------------------------------------------------------
# _strip_quotes
# ---------------------------------------------------------------------------


class TestStripQuotes:
    @pytest.mark.parametrize("value", ["'exp_a'", '"exp_a"', "exp_a", " exp_a "])
    def test_strips_surrounding_quotes_and_whitespace(self, value):
        assert _strip_quotes(value) == "exp_a"

    def test_inner_quotes_kept(self):
        assert _strip_quotes("a'b") == "a'b"

    def test_mismatched_quotes_kept(self):
        assert _strip_quotes("'exp_a\"") == "'exp_a\""


# ---------------------------------------------------------------------------
# _parse_int_or_none
# ---------------------------------------------------------------------------


class TestParseIntOrNone:
    @pytest.mark.parametrize("value", ["none", "None", "NONE", "null", "Null", "NULL"])
    def test_none_variants_return_none(self, value):
        assert _parse_int_or_none(value) is None

    def test_python_none_returns_none(self):
        assert _parse_int_or_none(None) is None

    def test_positive_int(self):
        assert _parse_int_or_none("7") == 7

    def test_zero(self):
        assert _parse_int_or_none("0") == 0

    def test_negative_int(self):
        assert _parse_int_or_none("-5") == -5

    def test_returns_int_type(self):
        assert isinstance(_parse_int_or_none("42"), int)

    def test_float_string_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_or_none("3.14")

    def test_non_numeric_raises(self):
        with pytest.raises((ValueError, TypeError)):
            _parse_int_or_none("abc")


# ---------------------------------------------------------------------------
# run_experiment — JobBatch resolution
# ---------------------------------------------------------------------------


def _save_minimal_batch(path, *, validation_expectation=None, experiment_protocol=None, preset=None) -> None:
    """Write a one-experiment, one-dataset JobBatch to `path` (bound to the suite `preset` when given)."""
    import pandas as pd
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment, Job, JobBatch, ValidationProtocol
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    experiment = AGModelBagExperiment(
        name="exp_a",
        model_cls=LGBModel,
        model_hyperparameters={},
        validation_protocol=experiment_protocol
        if experiment_protocol is not None
        else ValidationProtocol.custom(num_bag_folds=2),
        time_limit=60,
    )
    collection = TaskMetadataCollection.from_legacy_df(
        pd.DataFrame(
            {
                "tid": [1],
                "dataset": ["ds_a"],
                "problem_type": ["binary"],
                "n_folds": [1],
                "n_repeats": [1],
                "n_features": [5],
                "n_classes": [2],
                "NumberOfInstances": [100],
                "n_samples_train_per_fold": [80.0],
                "n_samples_test_per_fold": [20.0],
            },
        ),
    )
    if preset is not None:
        collection = collection.with_preset(preset)
    JobBatch(
        jobs=[Job.create(experiment, "ds_a", fold=0)],
        task_metadata=collection,
        validation_expectation=validation_expectation,
    ).save(path)


class TestRunExperimentResolution:
    def test_unknown_experiment_name_raises_with_available_names(self, tmp_path):
        batch_dir = tmp_path / "batch"
        _save_minimal_batch(batch_dir)
        with pytest.raises(ValueError, match="exp_a"):
            run_experiment(
                job_batch_dir=str(batch_dir),
                experiment_name="not_in_batch",
                dataset="ds_a",
                fold=0,
                repeat=0,
                output_dir=str(tmp_path / "out"),
                ignore_cache=False,
            )

    def test_coordinates_not_in_batch_raise(self, tmp_path):
        """A known experiment with coordinates that name no serialized job fails loudly
        (e.g. a stale job JSON against a regenerated batch).
        """
        batch_dir = tmp_path / "batch"
        _save_minimal_batch(batch_dir)
        for dataset, fold in [("not_a_dataset", 0), ("ds_a", 7)]:
            with pytest.raises(ValueError, match="not a job of the batch"):
                run_experiment(
                    job_batch_dir=str(batch_dir),
                    experiment_name="exp_a",
                    dataset=dataset,
                    fold=fold,
                    repeat=0,
                    output_dir=str(tmp_path / "out"),
                    ignore_cache=False,
                )


# ---------------------------------------------------------------------------
# setup_slurm_job  (no Ray, so returns None immediately)
# ---------------------------------------------------------------------------


class TestSetupSlurmJob:
    def test_returns_none_without_ray(self):
        result = setup_slurm_job(
            num_cpus=1,
            num_gpus=0,
            memory_limit=4,
            setup_ray_for_slurm_shared_resources_environment=False,
        )
        assert result is None

    def test_no_ray_setup_skips_ray(self, capsys):
        result = setup_slurm_job(
            num_cpus=2,
            num_gpus=0,
            memory_limit=8,
            setup_ray_for_slurm_shared_resources_environment=False,
        )
        assert result is None
        captured = capsys.readouterr()
        # Should NOT mention Ray setup when skipping it.
        assert "Ray" not in captured.out


class TestRunExperimentAppliesCacheConfig:
    """The runner applies the JobBatch's embedded CacheConfig before fitting (no CLI wiring)."""

    def test_applies_batch_cache_config(self, monkeypatch, tmp_path):
        import types

        import tabarena.benchmark.experiment as exp_mod
        from tabarena.caching import CacheConfig
        from tabarena.loaders import get_tabarena_cache_root, set_tabarena_cache_root

        job = types.SimpleNamespace(
            experiment=types.SimpleNamespace(name="exp"),
            task=types.SimpleNamespace(as_triple=lambda: ("ds", 0, 0)),
        )
        fake_batch = types.SimpleNamespace(
            jobs=[job],
            task_metadata=object(),
            cache_config=CacheConfig(tabarena=tmp_path / "tab"),
        )
        monkeypatch.setattr(exp_mod, "JobBatch", types.SimpleNamespace(load=lambda _dir: fake_batch))

        class _FakeRunner:
            def __init__(self, **kwargs):
                pass

            def run_jobs(self, jobs):
                return [{"metric_error": 0.1}]

        monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _FakeRunner)
        try:
            out = run_experiment(
                job_batch_dir="x",
                experiment_name="exp",
                dataset="ds",
                fold=0,
                repeat=0,
                output_dir=str(tmp_path / "out"),
                ignore_cache=False,
            )
            assert out == [{"metric_error": 0.1}]
            assert get_tabarena_cache_root() == tmp_path / "tab"  # batch's cache_config was applied
        finally:
            set_tabarena_cache_root(None)


class TestRunExperimentValidationExpectation:
    """The worker re-checks the experiment against the batch's validation expectation before fitting."""

    @staticmethod
    def _run(batch_dir, tmp_path):
        return run_experiment(
            job_batch_dir=str(batch_dir),
            experiment_name="exp_a",
            dataset="ds_a",
            fold=0,
            repeat=0,
            output_dir=str(tmp_path / "out"),
            ignore_cache=False,
        )

    @staticmethod
    def _fake_runner(monkeypatch):
        import tabarena.benchmark.experiment as exp_mod

        class _FakeRunner:
            def __init__(self, **kwargs):
                pass

            def run_jobs(self, jobs):
                return [{"metric_error": 0.1}]

        monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _FakeRunner)

    def test_a_foreign_experiment_in_an_enforced_batch_is_refused(self, tmp_path, monkeypatch):
        from tabarena.benchmark.experiment import ValidationExpectation, ValidationProtocol
        from tabarena.benchmark.validation_protocol import ValidationProtocolError

        self._fake_runner(monkeypatch)
        batch_dir = tmp_path / "batch"
        expectation = ValidationExpectation(protocol=ValidationProtocol(), enforced=True, arena="TabArena")
        _save_minimal_batch(batch_dir, validation_expectation=expectation)  # experiment runs 2x1, arena wants 8x1
        with pytest.raises(ValidationProtocolError, match="exp_a"):
            self._run(batch_dir, tmp_path)

    def test_a_matching_experiment_in_an_enforced_batch_runs(self, tmp_path, monkeypatch):
        from tabarena.benchmark.experiment import ValidationExpectation, ValidationProtocol

        self._fake_runner(monkeypatch)
        batch_dir = tmp_path / "batch"
        protocol = ValidationProtocol().with_origin(arena="TabArena", enforced=True)
        expectation = ValidationExpectation(protocol=protocol, enforced=True, arena="TabArena")
        _save_minimal_batch(batch_dir, validation_expectation=expectation, experiment_protocol=protocol)
        assert self._run(batch_dir, tmp_path) == [{"metric_error": 0.1}]

    def test_an_enforced_stamp_without_expectation_is_refused(self, tmp_path, monkeypatch):
        from tabarena.benchmark.experiment import ValidationProtocol
        from tabarena.benchmark.validation_protocol import ValidationProtocolError

        self._fake_runner(monkeypatch)
        batch_dir = tmp_path / "batch"
        stamped = ValidationProtocol().with_origin(arena="TabArena", enforced=True)
        _save_minimal_batch(batch_dir, experiment_protocol=stamped)  # no validation_protocol.json in the batch
        with pytest.raises(ValidationProtocolError, match="no enforced expectation"):
            self._run(batch_dir, tmp_path)

    def test_an_opted_out_batch_runs_any_protocol(self, tmp_path, monkeypatch):
        from tabarena.benchmark.experiment import ValidationExpectation, ValidationProtocol

        self._fake_runner(monkeypatch)
        batch_dir = tmp_path / "batch"
        expectation = ValidationExpectation(protocol=ValidationProtocol(), enforced=False, arena="TabArena")
        _save_minimal_batch(batch_dir, validation_expectation=expectation)
        assert self._run(batch_dir, tmp_path) == [{"metric_error": 0.1}]


# ---------------------------------------------------------------------------
# resolve_setup_ray  (Ray only when the experiment's fit can reach it)
# ---------------------------------------------------------------------------


def _in_memory_batch(experiment):
    import pandas as pd

    from tabarena.benchmark.experiment import Job, JobBatch
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    collection = TaskMetadataCollection.from_legacy_df(
        pd.DataFrame(
            {
                "tid": [1],
                "dataset": ["ds_a"],
                "problem_type": ["binary"],
                "n_folds": [1],
                "n_repeats": [1],
                "n_features": [5],
                "n_classes": [2],
                "NumberOfInstances": [100],
                "n_samples_train_per_fold": [80.0],
                "n_samples_test_per_fold": [20.0],
            },
        ),
    )
    job = Job.create(experiment, "ds_a", fold=0)
    return JobBatch(jobs=[job], task_metadata=collection), job


class TestResolveSetupRay:
    def test_sequential_local_bag_does_not_start_ray(self):
        from autogluon.tabular.models import LGBModel

        from tabarena.benchmark.experiment import AGModelBagExperiment, ValidationProtocol
        from tabflow_slurm.run_tabarena_experiment import job_problem_type, resolve_setup_ray

        experiment = AGModelBagExperiment(
            name="exp_seq",
            model_cls=LGBModel,
            model_hyperparameters={"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}},
            validation_protocol=ValidationProtocol.custom(num_bag_folds=2),
            time_limit=60,
        )
        batch, job = _in_memory_batch(experiment)
        assert job_problem_type(batch, job) == "binary"
        assert resolve_setup_ray(True, batch, job) is False
        assert resolve_setup_ray(False, batch, job) is False

    def test_error_in_uses_ray_starts_ray(self):
        import types

        from tabflow_slurm.run_tabarena_experiment import resolve_setup_ray

        def broken(**kwargs):
            raise RuntimeError("boom")

        job = types.SimpleNamespace(
            experiment=types.SimpleNamespace(uses_ray=broken),
            task=types.SimpleNamespace(dataset="ds_a"),
        )
        assert resolve_setup_ray(True, types.SimpleNamespace(task_metadata=None), job) is True


class TestRunExperimentWorkerFlags:
    """`cache_root` and `materialize_tasks` make the runner self-sufficient on a node without the shared filesystem."""

    @staticmethod
    def _fake_runner(monkeypatch):
        """Stub the fit; record the collection the runner was handed."""
        import tabarena.benchmark.experiment as exp_mod

        seen: dict = {}

        class _FakeRunner:
            def __init__(self, **kwargs):
                seen["task_metadata"] = kwargs["task_metadata"]

            def run_jobs(self, jobs):
                return [{"metric_error": 0.1}]

        monkeypatch.setattr(exp_mod, "ExperimentBatchRunner", _FakeRunner)
        return seen

    @staticmethod
    def _run(batch_dir, tmp_path, **kwargs):
        return run_experiment(
            job_batch_dir=str(batch_dir),
            experiment_name="exp_a",
            dataset="ds_a",
            fold=0,
            repeat=0,
            output_dir=str(tmp_path / "out"),
            ignore_cache=False,
            **kwargs,
        )

    def test_cache_root_overrides_the_batch_cache_config(self, monkeypatch, tmp_path):
        import types

        import openml

        import tabarena.benchmark.experiment as exp_mod
        from tabarena.caching import CacheConfig
        from tabarena.loaders import get_tabarena_cache_root, set_tabarena_cache_root

        job = types.SimpleNamespace(
            experiment=types.SimpleNamespace(name="exp_a"),
            task=types.SimpleNamespace(as_triple=lambda: ("ds_a", 0, 0)),
        )
        fake_batch = types.SimpleNamespace(
            jobs=[job], task_metadata=object(), cache_config=CacheConfig(tabarena=tmp_path / "head_node_tab")
        )
        monkeypatch.setattr(exp_mod, "JobBatch", types.SimpleNamespace(load=lambda _dir: fake_batch))
        self._fake_runner(monkeypatch)
        # setenv (not delenv) so monkeypatch records the original state and restores it afterwards.
        for name in ("HF_HOME", "DATA_FOUNDRY_CACHE", "HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
            monkeypatch.setenv(name, "placeholder")
        saved_openml_root = openml.config._root_cache_directory
        try:
            self._run("x", tmp_path, cache_root=str(tmp_path / "root"))
            # Every cache lives under the local root; the batch's head-node path was ignored.
            assert get_tabarena_cache_root() == tmp_path / "root" / "tabarena"
            assert os.environ["HF_HOME"] == str(tmp_path / "root" / "huggingface")
            assert os.environ["DATA_FOUNDRY_CACHE"] == str(tmp_path / "root" / "data_foundry")
            assert str(openml.config._root_cache_directory) == str(tmp_path / "root" / "openml")
        finally:
            set_tabarena_cache_root(None)
            openml.config.set_root_cache_directory(str(saved_openml_root))

    def test_without_the_flags_the_batch_collection_is_used_as_is(self, monkeypatch, tmp_path):
        seen = self._fake_runner(monkeypatch)
        batch_dir = tmp_path / "batch"
        _save_minimal_batch(batch_dir, preset="TabArena-v0.1")
        self._run(batch_dir, tmp_path)
        assert seen["task_metadata"].preset == "TabArena-v0.1"
        assert seen["task_metadata"].dataset_fold_repeats() == [("ds_a", 0, 0)]

    def test_materialize_tasks_downloads_through_the_recorded_suite(self, monkeypatch, tmp_path):
        from tabarena.benchmark.task.metadata import TaskMetadataCollection

        seen = self._fake_runner(monkeypatch)
        materialized: list = []
        monkeypatch.setattr(TaskMetadataCollection, "materialize", lambda self: materialized.append(self) or self)
        batch_dir = tmp_path / "batch"
        _save_minimal_batch(batch_dir, preset="TabArena-v0.1")
        self._run(batch_dir, tmp_path, materialize_tasks=True)
        # Exactly the job's split was materialized, through the suite's (OpenML) source, and the runner
        # resolves against that job-scoped collection.
        assert len(materialized) == 1
        assert materialized[0].dataset_fold_repeats() == [("ds_a", 0, 0)]
        assert materialized[0].preset == "TabArena-v0.1"
        assert seen["task_metadata"] is materialized[0]

    @staticmethod
    def _fake_batch_with_task(monkeypatch, *, task_id_str, data_foundry_uri, preset=None):
        """A batch whose job-scoped collection holds one task with the given id / uri and no real source."""
        import types

        import tabarena.benchmark.experiment as exp_mod

        job = types.SimpleNamespace(
            experiment=types.SimpleNamespace(name="exp_a"),
            task=types.SimpleNamespace(as_triple=lambda: ("ds_a", 0, 0)),
        )
        task = types.SimpleNamespace(
            task_id_str=task_id_str, data_foundry_uri=data_foundry_uri, tabarena_task_name="ds_a"
        )

        class _Scoped:
            def __init__(self):
                self.preset = preset
                self.materialized = 0

            def __iter__(self):
                return iter([task])

            def materialize(self):
                self.materialized += 1
                return self

        scoped = _Scoped()
        batch = types.SimpleNamespace(
            jobs=[job],
            task_metadata=types.SimpleNamespace(subset_to_jobs=lambda jobs: scoped),
            cache_config=None,
        )
        monkeypatch.setattr(exp_mod, "JobBatch", types.SimpleNamespace(load=lambda _dir: batch))
        return scoped

    def test_materialize_tasks_refuses_a_data_foundry_task_without_a_recorded_suite(self, monkeypatch, tmp_path):
        self._fake_runner(monkeypatch)
        self._fake_batch_with_task(monkeypatch, task_id_str="UserTask|1|ds_a/uuid", data_foundry_uri="ds_a/uuid")
        with pytest.raises(ValueError, match="task_source.json"):
            self._run("x", tmp_path, materialize_tasks=True)

    def test_materialize_tasks_without_a_suite_lets_openml_tasks_load_lazily(self, monkeypatch, tmp_path):
        seen = self._fake_runner(monkeypatch)
        scoped = self._fake_batch_with_task(monkeypatch, task_id_str="359955", data_foundry_uri=None)
        self._run("x", tmp_path, materialize_tasks=True)
        assert scoped.materialized == 0
        assert seen["task_metadata"] is scoped

    def test_materialize_tasks_rejects_a_task_id_that_embeds_a_cache_path(self, monkeypatch, tmp_path):
        self._fake_runner(monkeypatch)
        self._fake_batch_with_task(
            monkeypatch, task_id_str="UserTask|1|ds_a|/home/head/.cache/openml/tabarena_tasks", data_foundry_uri=None
        )
        with pytest.raises(ValueError, match="embeds a cache path"):
            self._run("x", tmp_path, materialize_tasks=True)
