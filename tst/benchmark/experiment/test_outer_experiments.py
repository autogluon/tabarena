"""Tests for the no-validation (outer) experiment path and shared preprocessing resolution.

Covers the pieces that let a model run with no train/val split while sharing the
``tabarena_default`` preprocessing with the bagged path: the pipeline resolver, ``AGModelWrapper``
resolving a pipeline, ``generate_outer_experiments``, and the bundle's ``outer_experiments`` mode.
No models are fit here (construction/resolution only).
"""

from __future__ import annotations

import pytest
from autogluon.core.models import AbstractModel
from autogluon.features import AutoMLPipelineFeatureGenerator

from tabarena.benchmark.exec_models.autogluon import AGModelWrapper
from tabarena.benchmark.experiment import AGModelOuterExperiment, BeyondArenaExperimentBundle
from tabarena.benchmark.preprocessing import (
    TabArenaModelAgnosticPreprocessing,
    resolve_preprocessing_pipeline,
)
from tabarena.benchmark.preprocessing.model_specific_default_preprocessing import (
    TabArenaModelSpecificPreprocessing,
)
from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.utils.config_utils import ConfigGenerator, generate_outer_experiments

MS_KEY = TabArenaModelSpecificPreprocessing.hp_key_kwargs


class _DummyModel(AbstractModel):
    ag_key = "DUMMYTESTMODEL"
    ag_name = "DummyTestModel"


class _FakeGroupedTask:
    """Minimal task exposing only the group columns a run-time injection needs."""

    def __init__(self, group_on=None, group_labels=None, group_time_on=None):
        self._vm = ValidationMetadata(
            target_name="t",
            group_on=group_on,
            group_labels=group_labels,
            group_time_on=group_time_on,
        )

    def get_validation_metadata(self) -> ValidationMetadata:
        return self._vm


class TestResolvePreprocessingPipeline:
    def test_default_is_automl_no_model_specific(self):
        pipeline = resolve_preprocessing_pipeline(None)
        assert pipeline.feature_generator_cls is AutoMLPipelineFeatureGenerator
        assert pipeline.apply_model_specific({"a": 1}) == {"a": 1}

    def test_tabarena_default(self):
        pipeline = resolve_preprocessing_pipeline("tabarena_default")
        assert pipeline.feature_generator_cls is TabArenaModelAgnosticPreprocessing
        assert MS_KEY in pipeline.apply_model_specific({})

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="not recognized"):
            resolve_preprocessing_pipeline("nope")


class TestAGModelWrapperPipeline:
    def _wrapper(self, pipeline: str | None) -> AGModelWrapper:
        return AGModelWrapper(
            model_cls=_DummyModel,
            hyperparameters={},
            preprocessing_pipeline=pipeline,
            problem_type="binary",
            eval_metric=None,
        )

    def test_tabarena_default_resolves_generator_and_model_specific(self):
        wrapper = self._wrapper("tabarena_default")
        assert type(wrapper._make_feature_generator()) is TabArenaModelAgnosticPreprocessing
        assert MS_KEY in wrapper.hyperparameters

    def test_default_keeps_automl_and_no_model_specific(self):
        wrapper = self._wrapper(None)
        assert type(wrapper._make_feature_generator()) is AutoMLPipelineFeatureGenerator
        assert MS_KEY not in wrapper.hyperparameters

    def test_fit_kwargs_stored_for_model_fit(self):
        wrapper = AGModelWrapper(
            model_cls=_DummyModel,
            hyperparameters={},
            fit_kwargs={"num_cpus": 1, "num_gpus": 0, "time_limit": 60},
            problem_type="binary",
            eval_metric=None,
        )
        assert wrapper.fit_kwargs == {"num_cpus": 1, "num_gpus": 0, "time_limit": 60}

    def test_verbosity_is_an_accepted_config_attr(self):
        # `verbosity` controls the feature generator's logging; default None leaves it untouched.
        assert AGModelWrapper(model_cls=_DummyModel, problem_type="binary", eval_metric=None).verbosity is None
        wrapper = AGModelWrapper(model_cls=_DummyModel, verbosity=2, problem_type="binary", eval_metric=None)
        assert wrapper.verbosity == 2


class TestOuterGroupMetadata:
    """An outer (AGModelWrapper) experiment sources group columns from the task at run time."""

    def test_group_cols_injected_for_outer_experiment(self):
        exp = AGModelOuterExperiment(
            name="DummyTestModel_c1",
            model_cls=_DummyModel,
            model_hyperparameters={},
            preprocessing_pipeline="tabarena_default",
        )
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        assert method_kwargs["group_cols"] == "grp"

    def test_user_set_group_cols_preserved(self):
        exp = AGModelOuterExperiment(
            name="DummyTestModel_c1",
            model_cls=_DummyModel,
            model_hyperparameters={},
            method_kwargs={"group_cols": "explicit"},
        )
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        assert method_kwargs["group_cols"] == "explicit"

    def test_no_injection_for_bagged_experiment(self):
        from autogluon.tabular.models import LGBModel

        from tabarena.benchmark.experiment import AGModelBagExperiment

        exp = AGModelBagExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={}, num_bag_folds=2)
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        assert "group_cols" not in method_kwargs


class TestGenerateOuterExperiments:
    def test_strips_predictor_keys_and_names_from_ag_args(self):
        configs = [{"ag_args": {"name_suffix": "_c1"}, "ag_args_ensemble": {"model_random_seed": 0}}]
        experiments = generate_outer_experiments(
            _DummyModel,
            configs,
            name_suffix_from_ag_args=True,
            preprocessing_pipeline="tabarena_default",
        )
        assert len(experiments) == 1
        exp = experiments[0]
        assert isinstance(exp, AGModelOuterExperiment)
        assert exp.name == "DummyTestModel_c1"
        hyperparameters = exp.method_kwargs["hyperparameters"]
        assert "ag_args" not in hyperparameters
        assert "ag_args_ensemble" not in hyperparameters
        assert exp.method_kwargs["preprocessing_pipeline"] == "tabarena_default"

    def test_merges_extra_model_hyperparameters(self):
        experiments = generate_outer_experiments(
            _DummyModel,
            [{"ag_args": {"name_suffix": "_c1"}}],
            name_suffix_from_ag_args=True,
            extra_model_hyperparameters={"ag.verbosity": 4},
        )
        assert experiments[0].method_kwargs["hyperparameters"]["ag.verbosity"] == 4


class TestBundleOuterMode:
    def test_emits_outer_experiments_with_pipeline_and_shuffle(self):
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        bundle = BeyondArenaExperimentBundle(models=[(generator, 0)], outer_experiments=True)
        experiments = bundle.build_experiments()
        assert len(experiments) == 1
        exp = experiments[0]
        assert isinstance(exp, AGModelOuterExperiment)
        # The suite's preprocessing + shuffle_features still apply; only validation/bagging is dropped.
        assert exp.method_kwargs["preprocessing_pipeline"] == "tabarena_default"
        assert exp.method_kwargs["shuffle_features"] is True
        # Resources + time_limit are forwarded to the model's fit.
        fit_kwargs = exp.method_kwargs["fit_kwargs"]
        assert "num_cpus" in fit_kwargs and "num_gpus" in fit_kwargs
        assert fit_kwargs["time_limit"] == bundle.DEFAULT_TIME_LIMIT
        # The bundle's verbosity drives the feature generator's output for outer fits.
        assert exp.method_kwargs["verbosity"] == bundle.verbosity

    def test_bagged_mode_is_unaffected(self):
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        bundle = BeyondArenaExperimentBundle(models=[(generator, 0)])  # default: bagged
        exp = bundle.build_experiments()[0]
        assert not isinstance(exp, AGModelOuterExperiment)
        assert exp.name.endswith("_BAG_L1")
