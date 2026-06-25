from __future__ import annotations

import argparse

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


def _save_minimal_batch(path) -> None:
    """Write a one-experiment, one-dataset JobBatch to `path`."""
    import pandas as pd
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment, Job, JobBatch
    from tabarena.benchmark.task.metadata import TaskMetadataCollection

    experiment = AGModelBagExperiment(
        name="exp_a",
        model_cls=LGBModel,
        model_hyperparameters={},
        num_bag_folds=2,
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
    JobBatch(jobs=[Job.create(experiment, "ds_a", fold=0)], task_metadata=collection).save(path)


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
