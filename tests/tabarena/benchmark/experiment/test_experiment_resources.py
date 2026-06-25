"""Tests for Experiment lazy compute-resource auto-detection and debug fold fitting."""

from __future__ import annotations

from autogluon.tabular.models import LGBModel

from tabarena.benchmark.experiment import AGExperiment, AGModelBagExperiment
from tabarena.benchmark.experiment.experiment_constructor import Experiment


def test_apply_resources_noop_when_concrete():
    mk = {"fit_kwargs": {"num_cpus": 4, "num_gpus": 0, "memory_limit": 16}}
    out = Experiment._apply_resources(mk)
    assert out is mk  # unchanged, no copy made


def test_apply_resources_no_fit_kwargs_is_noop():
    mk = {}
    assert Experiment._apply_resources(mk) is mk


def test_apply_resources_fills_none_without_mutating_original():
    mk = {"fit_kwargs": {"num_cpus": None, "num_gpus": 0, "memory_limit": None}}
    out = Experiment._apply_resources(mk)

    assert out is not mk  # a copy is returned
    assert isinstance(out["fit_kwargs"]["num_cpus"], int)
    assert out["fit_kwargs"]["num_cpus"] > 0
    assert isinstance(out["fit_kwargs"]["memory_limit"], int)
    assert out["fit_kwargs"]["memory_limit"] > 0
    assert out["fit_kwargs"]["num_gpus"] == 0
    # original left untouched (still requests auto-detection)
    assert mk["fit_kwargs"]["num_cpus"] is None
    assert mk["fit_kwargs"]["memory_limit"] is None


def test_apply_resources_detects_num_gpus_when_none():
    mk = {"fit_kwargs": {"num_cpus": 4, "num_gpus": None, "memory_limit": 16}}
    out = Experiment._apply_resources(mk)
    assert out is not mk
    assert isinstance(out["fit_kwargs"]["num_gpus"], int)
    assert out["fit_kwargs"]["num_gpus"] >= 0


def _bag_experiment(model_hyperparameters: dict | None = None) -> AGModelBagExperiment:
    return AGModelBagExperiment(
        name="lgbm",
        model_cls=LGBModel,
        model_hyperparameters=model_hyperparameters or {},
        num_bag_folds=2,
    )


def test_debug_fold_fitting_injects_sequential_local_for_bagged_model():
    out = _bag_experiment()._apply_debug_fold_fitting({"model_hyperparameters": {}})
    assert out["model_hyperparameters"]["ag_args_ensemble"]["fold_fitting_strategy"] == "sequential_local"


def test_debug_fold_fitting_preserves_explicit_strategy():
    mk = {"model_hyperparameters": {"ag_args_ensemble": {"fold_fitting_strategy": "parallel_local"}}}
    out = _bag_experiment()._apply_debug_fold_fitting(mk)
    assert out["model_hyperparameters"]["ag_args_ensemble"]["fold_fitting_strategy"] == "parallel_local"


def test_debug_fold_fitting_noop_for_non_bagged_experiment():
    # A full-predictor AGExperiment is not an AGSingleBagWrapper -> left untouched.
    exp = AGExperiment(name="ag", fit_kwargs={"hyperparameters": {"GBM": {}}})
    mk = {"model_hyperparameters": {}}
    out = exp._apply_debug_fold_fitting(mk)
    assert "ag_args_ensemble" not in out["model_hyperparameters"]


def test_init_method_kwargs_applies_debug_fold_fitting_only_when_debug():
    from tabarena.benchmark.task.metadata import ValidationMetadata

    class _Task:
        # Validation metadata is now requested for every experiment (injected uniformly as
        # read-only data); the dynamic protocol is still off here, so it is not acted upon.
        def get_validation_metadata(self):
            return ValidationMetadata()

    exp = _bag_experiment()
    assert (
        exp.init_method_kwargs(task=_Task(), debug_mode=False)["model_hyperparameters"]
        .get("ag_args_ensemble", {})
        .get("fold_fitting_strategy")
        is None
    )
    assert (
        exp.init_method_kwargs(task=_Task(), debug_mode=True)["model_hyperparameters"]["ag_args_ensemble"][
            "fold_fitting_strategy"
        ]
        == "sequential_local"
    )
