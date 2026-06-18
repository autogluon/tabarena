"""Tests for the holdout experiment path: a single ``TabularPredictor`` fit with a real
train/val split but no bagging or weighted ensemble.

Covers the generator (:func:`generate_holdout_experiments` /
``AGConfigGenerator.generate_all_holdout_experiments``) and the bundle's ``holdout_experiments``
mode, contrasting it with the bagged default. No models are fit here (construction only).
"""

from __future__ import annotations

import pytest
from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import (
    AGModelBagExperiment,
    AGModelExperiment,
    AGModelOuterExperiment,
    BeyondArenaExperimentBundle,
)
from tabarena.utils.config_utils import ConfigGenerator, generate_holdout_experiments


class _DummyModel(AbstractModel):
    ag_key = "DUMMYTESTMODEL"
    ag_name = "DummyTestModel"


class TestGenerateHoldoutExperiments:
    def test_builds_single_model_experiment_without_bag_suffix(self):
        configs = [{"ag_args": {"name_suffix": "_c1"}}]
        experiments = generate_holdout_experiments(
            _DummyModel,
            configs,
            name_suffix_from_ag_args=True,
            preprocessing_pipeline="tabarena_default",
            dynamic_tabarena_validation_protocol=True,
        )
        assert len(experiments) == 1
        exp = experiments[0]
        assert isinstance(exp, AGModelExperiment)
        # A holdout fit is a single (non-bagged) model, tagged `_HOLDOUT` (mirrors bagged `_BAG_L1`).
        assert not isinstance(exp, (AGModelBagExperiment, AGModelOuterExperiment))
        assert exp.name == "DummyTestModel_c1_HOLDOUT"
        assert exp.preprocessing_pipeline == "tabarena_default"
        assert exp.dynamic_tabarena_validation_protocol is True

    def test_keeps_ag_args_in_model_hyperparameters(self):
        # Unlike the outer flavour, holdout goes through `TabularPredictor`, which consumes
        # `ag_args` (e.g. for naming), so it is kept in the model hyperparameters.
        experiments = generate_holdout_experiments(
            _DummyModel,
            [{"ag_args": {"name_suffix": "_c1"}}],
            name_suffix_from_ag_args=True,
        )
        assert "ag_args" in experiments[0].method_kwargs["model_hyperparameters"]

    def test_merges_extra_model_hyperparameters(self):
        experiments = generate_holdout_experiments(
            _DummyModel,
            [{"ag_args": {"name_suffix": "_c1"}}],
            name_suffix_from_ag_args=True,
            extra_model_hyperparameters={"ag.verbosity": 4},
        )
        assert experiments[0].method_kwargs["model_hyperparameters"]["ag.verbosity"] == 4

    def test_time_limit_injected_as_model_max_time_limit(self):
        experiments = generate_holdout_experiments(
            _DummyModel,
            [{"ag_args": {"name_suffix": "_c1"}}],
            name_suffix_from_ag_args=True,
            time_limit=123,
        )
        # No bagging -> the limit rides on the model's top-level `ag.max_time_limit`.
        assert experiments[0].method_kwargs["model_hyperparameters"]["ag.max_time_limit"] == 123


class TestBundleHoldoutMode:
    def test_emits_holdout_experiments_with_pipeline_resources_and_protocol(self):
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        bundle = BeyondArenaExperimentBundle(models=[(generator, 0)], holdout_experiments=True)
        experiments = bundle.build_experiments()
        assert len(experiments) == 1
        exp = experiments[0]
        assert isinstance(exp, AGModelExperiment)
        assert not isinstance(exp, (AGModelBagExperiment, AGModelOuterExperiment))
        # Single model -> tagged `_HOLDOUT` rather than the bagged `_BAG_L1`.
        assert exp.name == "DummyTestModel_c1_HOLDOUT"
        # The bundle's preprocessing + shuffle_features + validation protocol all still apply.
        assert exp.preprocessing_pipeline == "tabarena_default"
        assert exp.dynamic_tabarena_validation_protocol is True
        assert exp.method_kwargs["shuffle_features"] is True
        # Compute resources are baked into the predictor fit kwargs (None == auto-detect at run time).
        fit_kwargs = exp.method_kwargs["fit_kwargs"]
        assert "num_cpus" in fit_kwargs and "num_gpus" in fit_kwargs and "memory_limit" in fit_kwargs
        # time_limit (no bagging) -> model-level `ag.max_time_limit`; model_verbosity merged in.
        model_hyperparameters = exp.method_kwargs["model_hyperparameters"]
        assert model_hyperparameters["ag.max_time_limit"] == bundle.DEFAULT_TIME_LIMIT
        assert model_hyperparameters["ag.verbosity"] == bundle.model_verbosity

    def test_bagged_mode_is_unaffected(self):
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        bundle = BeyondArenaExperimentBundle(models=[(generator, 0)])  # default: bagged
        exp = bundle.build_experiments()[0]
        assert isinstance(exp, AGModelBagExperiment)
        assert exp.name.endswith("_BAG_L1")

    def test_outer_and_holdout_are_mutually_exclusive(self):
        with pytest.raises(ValueError, match="mutually exclusive"):
            BeyondArenaExperimentBundle(outer_experiments=True, holdout_experiments=True)
