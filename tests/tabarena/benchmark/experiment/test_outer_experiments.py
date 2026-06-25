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
    """An outer (AGModelWrapper) experiment sources group columns from the task's validation metadata.

    Group columns reach every method via the uniform ``validation_metadata`` injection (the base
    attribute) rather than a separate ``group_cols`` key; the outer ``AGModelWrapper`` reads them
    straight from there.
    """

    @staticmethod
    def _spy_build_feature_generator(monkeypatch) -> dict:
        """Capture the group params ``AGModelWrapper`` forwards to ``build_feature_generator``."""
        captured: dict = {}

        def fake_build(cls, feature_generator_kwargs=None, *, group_cols=None, group_labels=None, group_time_on=None):
            captured.update(group_cols=group_cols, group_labels=group_labels, group_time_on=group_time_on)
            return object()

        monkeypatch.setattr("tabarena.benchmark.exec_models.autogluon.build_feature_generator", fake_build)
        return captured

    def test_validation_metadata_injected_for_outer_experiment(self):
        exp = AGModelOuterExperiment(
            name="DummyTestModel_c1",
            model_cls=_DummyModel,
            model_hyperparameters={},
            preprocessing_pipeline="tabarena_default",
        )
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        # The task metadata is injected uniformly as read-only data (not a separate group_cols key).
        assert method_kwargs["validation_metadata"].group_on == "grp"
        assert "group_cols" not in method_kwargs

    def test_outer_wrapper_resolves_group_from_validation_metadata(self, monkeypatch):
        captured = self._spy_build_feature_generator(monkeypatch)
        wrapper = AGModelWrapper(
            model_cls=_DummyModel,
            hyperparameters={},
            preprocessing_pipeline="tabarena_default",
            validation_metadata=ValidationMetadata(group_on="grp", group_labels="per_sample"),
            problem_type="binary",
            eval_metric=None,
        )
        wrapper._make_feature_generator()
        assert captured["group_cols"] == "grp"
        assert captured["group_labels"] == "per_sample"

    def test_bagged_receives_metadata_but_does_not_act(self):
        from autogluon.tabular.models import LGBModel

        from tabarena.benchmark.experiment import AGModelBagExperiment

        exp = AGModelBagExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={}, num_bag_folds=2)
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        # Bagged (dynamic protocol off): metadata is present as data, but the policy gate that makes
        # the wrapper *act* on it (use_task_specific_validation) is not set, and no group_cols key.
        assert method_kwargs["validation_metadata"].group_on == "grp"
        assert "use_task_specific_validation" not in method_kwargs
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


class _FakeTextTask:
    """Minimal task exposing only ``has_text``, which is all the cache scope inspects."""

    def __init__(self, *, has_text: bool):
        self.has_text = has_text


class TestTextCacheScope:
    """The text-embedding cache scope is independent of the validation protocol.

    A standalone ``Experiment`` defaults to ``text_cache_mode="off"``; when set to ``require`` it
    must load (and require) the cache for a text task even on the outer path, where
    ``dynamic_tabarena_validation_protocol=False`` (regression guard for the outer-path discrepancy).
    """

    def _outer_experiment(self, **kwargs) -> AGModelOuterExperiment:
        exp = AGModelOuterExperiment(
            name="DummyTestModel_c1",
            model_cls=_DummyModel,
            model_hyperparameters={},
            preprocessing_pipeline="tabarena_default",
            **kwargs,
        )
        assert exp.dynamic_tabarena_validation_protocol is False  # the outer path's default
        return exp

    def test_standalone_default_mode_is_off(self):
        assert self._outer_experiment().text_cache_mode == "off"

    def test_non_text_task_is_null_scope(self):
        from contextlib import nullcontext

        scope = self._outer_experiment(text_cache_mode="require").task_cache_scope(
            task=_FakeTextTask(has_text=False),
            cache_task_key="any-task",
        )
        assert isinstance(scope, nullcontext)

    def test_outer_text_task_requires_cache_when_required(self, monkeypatch):
        from pathlib import Path

        from tabarena.benchmark.preprocessing import text_cache

        # No cache on disk; keep the require-branch error message off the real filesystem.
        monkeypatch.setattr(text_cache, "resolve_existing_cache_path", lambda task_key: None)
        monkeypatch.setattr(text_cache, "text_cache_path", lambda task_key: Path(f"/nonexistent/{task_key}.parquet"))

        scope = self._outer_experiment(text_cache_mode="require").task_cache_scope(
            task=_FakeTextTask(has_text=True),
            cache_task_key="missing-task",
        )
        with pytest.raises(FileNotFoundError), scope:
            pass


class TestBuildPathFlagParity:
    """Bundle-built experiments enforce the bundle's text-cache mode on every build path.

    BeyondArena requires the cache across bagged / holdout / outer alike; the validation protocol
    stays bagged/holdout-only by design (outer has no train/val split). The v0.1 bundle disables
    the cache (it uses AutoGluon-default preprocessing and ships no semantic-text caches).
    """

    @staticmethod
    def _build(**bundle_kwargs) -> AGModelOuterExperiment:
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        return BeyondArenaExperimentBundle(models=[(generator, 0)], **bundle_kwargs).build_experiments()[0]

    def test_text_cache_required_across_build_paths(self):
        bagged = self._build()
        holdout = self._build(holdout_experiments=True)
        outer = self._build(outer_experiments=True)
        # The bundle enforces `require` regardless of build path.
        assert bagged.text_cache_mode == holdout.text_cache_mode == outer.text_cache_mode == "require"
        # The dynamic validation protocol stays bagged/holdout-only (outer = no validation split).
        assert bagged.dynamic_tabarena_validation_protocol is True
        assert holdout.dynamic_tabarena_validation_protocol is True
        assert outer.dynamic_tabarena_validation_protocol is False

    def test_v0pt1_bundle_disables_text_cache(self):
        from tabarena.benchmark.experiment.bundle import TabArenaV0pt1ExperimentBundle

        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        exp = TabArenaV0pt1ExperimentBundle(models=[(generator, 0)]).build_experiments()[0]
        assert exp.text_cache_mode == "off"


class TestOuterUnhonoredKnobs:
    """Outer fits warn for predictor-level build knobs they cannot honor (no silent drop)."""

    def _bundle(self):
        generator = ConfigGenerator(search_space={}, model_cls=_DummyModel, manual_configs=[{}])
        return BeyondArenaExperimentBundle(models=[(generator, 0)], outer_experiments=True)

    def test_warns_on_memory_limit(self):
        with pytest.warns(UserWarning, match="memory_limit"):
            self._bundle().build_experiments(memory_limit=8000)

    def test_warns_on_time_limit_with_preprocessing(self):
        with pytest.warns(UserWarning, match="time_limit_with_preprocessing"):
            self._bundle().build_experiments(time_limit_with_preprocessing=True)

    def test_no_warn_on_defaults(self):
        import warnings

        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            self._bundle().build_experiments()
        messages = [str(w.message) for w in caught]
        assert not any("memory_limit" in m or "time_limit_with_preprocessing" in m for m in messages)
