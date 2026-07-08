"""Tests for the external-system experiment path — a self-contained ML system (not a single AutoGluon model).

Covers the wiring that lets a self-contained *system* run through the same hub as any other
method: the ``ExternalSystemModel`` base contract, ``SystemConfigGenerator``, the bundle's
``system_experiments`` mode, run-time ``validation_metadata`` injection, YAML round-trip, and a
small end-to-end fit. Uses a trivial in-test system (no AutoGluon) so it stays fast; the runnable
AutoGluon-backed demo lives in ``examples/advanced/run_quickstart_tabarena_external_system.py``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.benchmark.experiment import (
    ExternalSystemExperiment,
    TabArenaV0pt1ExperimentBundle,
    YamlExperimentSerializer,
)
from tabarena.benchmark.task.metadata import ValidationMetadata
from tabarena.utils.config_utils import ConfigGenerator, SystemConfigGenerator


class _DemoSystem(ExternalSystemModel):
    """Trivial ``ExternalSystemModel`` for tests (no AutoGluon): predicts class priors / target mean."""

    def __init__(self, *, note: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.note = note  # an accepted config param, to exercise the config -> init-kwarg flow
        self.seen: dict | None = None  # the context `_fit_system` received (for assertions)

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name,
        problem_type,
        eval_metric,
        validation_metadata,
        num_cpus,
        num_gpus,
        memory_limit,
        time_limit,
        random_state,
    ):
        self.seen = {
            "target_name": target_name,
            "problem_type": problem_type,
            "eval_metric": eval_metric,
            "validation_metadata": validation_metadata,
            "num_cpus": num_cpus,
            "num_gpus": num_gpus,
            "memory_limit": memory_limit,
            "time_limit": time_limit,
            "random_state": random_state,
        }
        # `X` is ours to edit in place (the base decided copy-vs-in-place) — append the target under
        # its semantic name (`target_name`), as a real system would.
        X[target_name] = y
        if problem_type == "regression":
            self._value = float(y.mean())
        else:
            self._classes = sorted(pd.unique(y))
            self._priors = y.value_counts(normalize=True)
        return self

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        return pd.Series(self._value, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        row = [float(self._priors.get(c, 0.0)) for c in self._classes]
        return pd.DataFrame([row] * len(X), columns=self._classes, index=X.index)


class _FakeGroupedTask:
    """Minimal task exposing only the validation metadata the run-time injection reads."""

    def __init__(self, group_on=None):
        self._vm = ValidationMetadata(target_name="t", group_on=group_on)

    def get_validation_metadata(self) -> ValidationMetadata:
        return self._vm


def _generator(manual_configs=None) -> SystemConfigGenerator:
    return SystemConfigGenerator(
        model_cls=_DemoSystem,
        name="DemoSystem",
        manual_configs=manual_configs if manual_configs is not None else [{}],
    )


class TestExternalSystemModelContract:
    """The base advertises the no-AutoGluon defaults and carries validation metadata uniformly."""

    def test_default_flags(self):
        assert ExternalSystemModel.preprocess_data is False
        assert ExternalSystemModel.preprocess_label is False
        assert ExternalSystemModel.can_get_oof is False
        assert ExternalSystemModel.can_get_error_val is False

    def test_validation_metadata_is_carried_from_base(self):
        model = _DemoSystem(
            problem_type="binary",
            eval_metric=None,
            validation_metadata=ValidationMetadata(target_name="t", group_on="g"),
        )
        assert model.validation_metadata.target_name == "t"
        assert model.validation_metadata.group_on == "g"

    def test_validation_metadata_defaults_to_empty(self):
        model = _DemoSystem(problem_type="binary", eval_metric=None)
        assert isinstance(model.validation_metadata, ValidationMetadata)
        assert model.validation_metadata.target_name is None

    def test_compute_resources_exposed_at_fit_time(self):
        model = _DemoSystem(
            problem_type="binary",
            eval_metric=None,
            fit_kwargs={"num_cpus": 4, "num_gpus": 1, "memory_limit": 8, "time_limit": 60},
        )
        assert (model.num_cpus, model.num_gpus, model.memory_limit, model.time_limit) == (4, 1, 8, 60)

    def test_compute_resources_default_to_none(self):
        model = _DemoSystem(problem_type="binary", eval_metric=None)
        assert model.num_cpus is None
        assert model.time_limit is None


class TestSystemConfigGenerator:
    def test_requires_explicit_name(self):
        with pytest.raises(AssertionError, match="explicit `name`"):
            SystemConfigGenerator(model_cls=_DemoSystem, name="")

    def test_builds_external_system_experiments(self):
        experiments = _generator(manual_configs=[{}, {"note": "x"}]).generate_all_system_experiments(
            num_random_configs=0
        )
        assert [type(e).__name__ for e in experiments] == ["ExternalSystemExperiment", "ExternalSystemExperiment"]
        assert [e.name for e in experiments] == ["DemoSystem_c1", "DemoSystem_c2"]
        # The config dict becomes the system's init kwargs.
        assert experiments[1].method_kwargs["note"] == "x"

    @pytest.mark.parametrize("flavour", ["bag", "holdout", "outer"])
    def test_autogluon_flavours_raise(self, flavour):
        gen = _generator()
        with pytest.raises(NotImplementedError, match="system_experiments=True"):
            getattr(gen, f"generate_all_{flavour}_experiments")(num_random_configs=0)


class TestExternalSystemExperimentInjection:
    """The experiment injects the task's validation metadata and forwards system hyperparameters."""

    def test_injects_validation_metadata_and_forwards_hyperparameters(self):
        exp = ExternalSystemExperiment(
            name="DemoSystem_c1",
            system_cls=_DemoSystem,
            system_hyperparameters={"note": "x"},
        )
        method_kwargs = exp.init_method_kwargs(task=_FakeGroupedTask(group_on="grp"))
        assert method_kwargs["validation_metadata"].group_on == "grp"
        assert method_kwargs["note"] == "x"
        # No-validation regime: the AutoGluon "act on it" policy gate is never set.
        assert "use_task_specific_validation" not in method_kwargs

    def test_yaml_round_trip_resolves_system_cls(self):
        # The base ``ExternalSystemModel`` is used here because YAML serializes ``system_cls`` as an
        # import path, and a test-local class is not importable by path (the demo subclasses are).
        experiments = SystemConfigGenerator(
            model_cls=ExternalSystemModel, name="DemoSystem", manual_configs=[{}, {}]
        ).generate_all_system_experiments(num_random_configs=0)
        restored = YamlExperimentSerializer.from_yaml_str(YamlExperimentSerializer.to_yaml_str(experiments))
        assert [type(e).__name__ for e in restored] == ["ExternalSystemExperiment", "ExternalSystemExperiment"]
        assert [e.name for e in restored] == [e.name for e in experiments]
        assert restored[0].method_cls is ExternalSystemModel


class TestBundleSystemMode:
    def test_emits_external_system_experiments(self):
        bundle = TabArenaV0pt1ExperimentBundle(
            models=[(_generator(manual_configs=[{}, {}]), 0)],
            system_experiments=True,
        )
        experiments = bundle.build_experiments()
        assert [type(e).__name__ for e in experiments] == ["ExternalSystemExperiment", "ExternalSystemExperiment"]
        # Same per-pipeline naming convention as the bag/outer paths (v0.1 tags the "default" pipeline).
        assert [e.name for e in experiments] == ["DemoSystem_c1_default", "DemoSystem_c2_default"]
        # The base ``shuffle_features`` capability is forwarded (v0.1 default: False); AutoGluon-only
        # knobs are not.
        assert experiments[0].method_kwargs["shuffle_features"] is False
        # Compute resources are forwarded the same way as for other methods (time_limit baked in,
        # num_cpus/num_gpus/memory_limit present for run-time auto-detection).
        fit_kwargs = experiments[0].method_kwargs["fit_kwargs"]
        assert fit_kwargs["time_limit"] == TabArenaV0pt1ExperimentBundle.DEFAULT_TIME_LIMIT
        assert {"num_cpus", "num_gpus", "memory_limit"} <= fit_kwargs.keys()

    def test_mutually_exclusive_with_other_modes(self):
        with pytest.raises(ValueError, match="mutually"):
            TabArenaV0pt1ExperimentBundle(models=[], system_experiments=True, outer_experiments=True)

    def test_requires_system_config_generator(self):
        from autogluon.tabular.models import LGBModel

        bundle = TabArenaV0pt1ExperimentBundle(
            models=[(ConfigGenerator(search_space={}, model_cls=LGBModel, manual_configs=[{}]), 0)],
            system_experiments=True,
        )
        with pytest.raises(TypeError, match="SystemConfigGenerator"):
            bundle.build_experiments()


class TestDemoSystemFit:
    """End-to-end fit through the base machinery with the trivial in-test system (no AutoGluon)."""

    @staticmethod
    def _features(n=40):
        rng = np.random.default_rng(0)
        return pd.DataFrame({"a": rng.normal(size=n), "b": rng.normal(size=n)}), rng

    def _model(self, problem_type: str) -> _DemoSystem:
        return _DemoSystem(
            problem_type=problem_type,
            eval_metric=None,
            validation_metadata=ValidationMetadata(target_name="t"),
        )

    def test_binary_predict_proba(self):
        X, rng = self._features()
        y = pd.Series(rng.integers(0, 2, size=len(X)))
        model = self._model("binary")
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert isinstance(proba, pd.DataFrame)
        assert proba.shape == (len(X), 2)
        assert list(proba.index) == list(X.index)

    def test_regression_predict(self):
        X, rng = self._features()
        y = pd.Series(rng.normal(size=len(X)))
        model = self._model("regression")
        model.fit(X, y)
        preds = model.predict(X)
        assert isinstance(preds, pd.Series)
        assert len(preds) == len(X)

    def test_fit_system_receives_full_context(self):
        X, rng = self._features()
        y = pd.Series(rng.integers(0, 2, size=len(X)))
        model = _DemoSystem(
            problem_type="binary",
            eval_metric="roc_auc",
            validation_metadata=ValidationMetadata(target_name="t", group_on="grp"),
            fit_kwargs={"num_cpus": 4, "num_gpus": 1, "memory_limit": 8, "time_limit": 60},
        )
        model.fit(X, y)
        # Everything the fit needs is passed to `_fit_system` (read off args, not `self`).
        assert model.seen["target_name"] == "t"  # the task's semantic target name, not a placeholder
        assert model.seen["problem_type"] == "binary"
        assert model.seen["eval_metric"] == "roc_auc"
        assert model.seen["validation_metadata"].group_on == "grp"
        assert (model.seen["num_cpus"], model.seen["num_gpus"]) == (4, 1)
        assert (model.seen["memory_limit"], model.seen["time_limit"]) == (8, 60)
        # No split seed is set on a direct `fit` (outside the runner), so `random_state` is None.
        assert model.seen["random_state"] is None

    def test_does_not_mutate_caller_frame_by_default(self):
        X, rng = self._features()
        y = pd.Series(rng.integers(0, 2, size=len(X)))
        self._model("binary").fit(X, y)
        # The system appended the target column to its copy, not to the caller's frame.
        assert "t" not in X.columns

    def test_edits_owned_frame_in_place(self):
        X, rng = self._features()
        y = pd.Series(rng.integers(0, 2, size=len(X)))
        model = self._model("binary")
        model._can_use_data_in_place = True  # as set by ``fit_custom`` when the task lazy-loads
        model.fit(X, y)
        # No copy made: the system edited the caller's frame directly (saves RAM on large data).
        assert "t" in X.columns
