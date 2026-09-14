"""Tests for the AutoGluon exec models beyond the persist bracket.

Fakes stand in for AutoGluon (shared ones in ``conftest.py``): metadata bookkeeping (the best
model's ``get_info()`` is collected once and read consistently, the new inference-side keys), the
single-child bag-artifact reuse of the timed prediction and its fallbacks, the lazy internal-data
load of ``get_per_child_val_idx``, the artifact-root environment override and ``uses_ray``. The
``models``-marked class at the end fits a tiny real RandomForest bag to prove the reuse path is
bit-identical to the per-child forward pass.
"""

from __future__ import annotations

import types
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from autogluon.core.data.label_cleaner import (
    LabelCleanerBinary,
    LabelCleanerDummy,
    LabelCleanerMulticlass,
    LabelCleanerMulticlassToBinary,
)
from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import (
    MODEL_ARTIFACTS_BASE_PATH_ENV,
    AGModelWrapper,
    AGSingleBagWrapper,
    AGSingleWrapper,
    AGWrapper,
    _hyperparameters_user_from_info,
)
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tests.tabarena.benchmark.exec_models.conftest import _FakeFittedModel


class _DummyModel(AbstractModel):
    ag_key = "DUMMYMETADATAMODEL"
    ag_name = "DummyMetadataModel"


class _FakePredictor:
    def __init__(self, model: _FakeFittedModel):
        self.model_best = "Dummy_BAG_L1"
        self._trainer = SimpleNamespace(load_model=lambda name: model, models={})
        self.load_model_calls = 0
        trainer_load = self._trainer.load_model

        def counted(name):
            assert name == self.model_best
            self.load_model_calls += 1
            return trainer_load(name)

        self._trainer.load_model = counted

    def model_names(self, can_infer: bool = False) -> list[str]:
        return [self.model_best]

    def model_hyperparameters(self, *args, **kwargs):
        raise AssertionError("get_metadata must read hyperparameters from the collected info")


BAGGED_INFO = {
    "hyperparameters_user": {"model_random_seed": 3},
    "bagged_info": {"child_hyperparameters_user": {"n_estimators": 8}},
}
BAGGED_INFO_NO_ENSEMBLE_ARGS = {
    "hyperparameters_user": {},
    "bagged_info": {"child_hyperparameters_user": {"n_estimators": 8}},
}
SINGLE_INFO = {"hyperparameters_user": {"n_estimators": 8}}


def _make_wrapper(model: _FakeFittedModel) -> AGSingleWrapper:
    wrapper = AGSingleWrapper(
        model_cls=_DummyModel,
        model_hyperparameters={"n_estimators": 8},
        problem_type="regression",
        eval_metric=get_metric("rmse", problem_type="regression"),
    )
    wrapper.predictor = _FakePredictor(model)
    return wrapper


class TestHyperparametersUserFromInfo:
    def test_bagged_appends_ensemble_args(self):
        assert _hyperparameters_user_from_info(BAGGED_INFO) == {
            "n_estimators": 8,
            "ag_args_ensemble": {"model_random_seed": 3},
        }

    def test_bagged_without_ensemble_args(self):
        assert _hyperparameters_user_from_info(BAGGED_INFO_NO_ENSEMBLE_ARGS) == {"n_estimators": 8}

    def test_bagged_does_not_mutate_info(self):
        info = {
            "hyperparameters_user": {"model_random_seed": 3},
            "bagged_info": {"child_hyperparameters_user": {"n_estimators": 8}},
        }
        _hyperparameters_user_from_info(info)
        assert info["bagged_info"]["child_hyperparameters_user"] == {"n_estimators": 8}

    def test_single_model(self):
        assert _hyperparameters_user_from_info(SINGLE_INFO) == {"n_estimators": 8}


class TestGetMetadata:
    def test_collects_info_once_and_shares_it(self):
        model = _FakeFittedModel(BAGGED_INFO)
        wrapper = _make_wrapper(model)

        metadata = wrapper.get_metadata()

        assert model.get_info_calls == 1
        assert wrapper.predictor.load_model_calls == 1
        assert metadata["info"] is BAGGED_INFO
        assert metadata["hyperparameters"] == {"n_estimators": 8, "ag_args_ensemble": {"model_random_seed": 3}}
        assert metadata["model_cls"] == "_DummyModel"
        assert metadata["model_type"] == "DUMMYMETADATAMODEL"
        assert metadata["disk_usage"] == 42
        assert metadata["num_cpus_child"] == 2
        assert metadata["fit_metadata"] == {"fit": "metadata"}
        assert metadata["memory_usage_estimate"] == 12345

    def test_parts_collect_info_themselves_when_not_given(self):
        model = _FakeFittedModel(SINGLE_INFO)
        wrapper = _make_wrapper(model)

        assert wrapper.get_hyperparameters() == {"n_estimators": 8}
        assert wrapper.get_metadata_init()["hyperparameters"] == {"n_estimators": 8}
        assert wrapper.get_metadata_fit()["info"] is SINGLE_INFO
        assert model.get_info_calls == 3

    def test_records_inference_side_keys(self):
        info = {
            **BAGGED_INFO,
            "children_info": {"S1F1": {"shared_weights": {"library": "lib"}}, "S1F2": {}},
        }
        model = _FakeFittedModel(info)
        wrapper = _make_wrapper(model)
        wrapper._persisted_models = ["Dummy_BAG_L1"]
        wrapper._prepared_models = ["Dummy_BAG_L1", "S1F1"]

        metadata = wrapper.get_metadata()

        assert metadata["persist"] is True
        assert metadata["persisted_models"] == ["Dummy_BAG_L1"]
        assert metadata["prepared_for_inference"] == ["Dummy_BAG_L1", "S1F1"]
        assert metadata["shared_weights"]["children"] == {"S1F1": {"library": "lib"}, "S1F2": None}
        assert set(metadata["shared_weights"]["registry"]) >= {"capacity", "stats", "entries"}
        assert metadata["per_child_test_source"] is None  # no bag artifact collected yet

    def test_single_model_has_no_shared_weights_children(self):
        metadata = _make_wrapper(_FakeFittedModel(SINGLE_INFO)).get_metadata()
        assert metadata["shared_weights"]["children"] == {}


# --- artifact root override -------------------------------------------------------------


class TestModelArtifactsBasePathOverride:
    @staticmethod
    def _frames():
        return pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0]}), pd.Series([1.0, 2.0, 3.0, 4.0])

    def test_env_replaces_default_base_path_on_the_fit_copy_only(self, monkeypatch, tmp_path):
        X, y = self._frames()
        wrapper = AGWrapper(
            init_kwargs={"default_base_path": "/configured/root"},
            problem_type="regression",
            eval_metric=get_metric("rmse", problem_type="regression"),
        )
        monkeypatch.setenv(MODEL_ARTIFACTS_BASE_PATH_ENV, str(tmp_path))
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert init_kwargs["default_base_path"] == str(tmp_path)
        assert wrapper.init_kwargs["default_base_path"] == "/configured/root"  # the configured kwargs are untouched

        monkeypatch.delenv(MODEL_ARTIFACTS_BASE_PATH_ENV)
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert init_kwargs["default_base_path"] == "/configured/root"

    def test_env_without_a_configured_base_path_adds_nothing(self, monkeypatch, tmp_path):
        X, y = self._frames()
        wrapper = AGWrapper(problem_type="regression", eval_metric=get_metric("rmse", problem_type="regression"))
        monkeypatch.setenv(MODEL_ARTIFACTS_BASE_PATH_ENV, str(tmp_path))
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert "default_base_path" not in init_kwargs

    def test_single_wrapper_metadata_keeps_the_configured_path(self, monkeypatch, tmp_path):
        X, y = self._frames()
        wrapper = AGSingleWrapper(
            model_cls=_DummyModel,
            model_hyperparameters={},
            init_kwargs={"default_base_path": "/configured/root"},
            problem_type="regression",
            eval_metric=get_metric("rmse", problem_type="regression"),
        )
        monkeypatch.setenv(MODEL_ARTIFACTS_BASE_PATH_ENV, str(tmp_path))
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert init_kwargs["default_base_path"] == str(tmp_path)
        assert wrapper.get_metadata_init(info=SINGLE_INFO)["init_kwargs_extra"] == {
            "default_base_path": "/configured/root"
        }


# --- uses_ray ---------------------------------------------------------------------------


class TestUsesRay:
    @staticmethod
    def _kwargs(*, fit_kwargs: dict | None = None, ag_args_ensemble: dict | None = None, model_cls=_DummyModel):
        hyperparameters = {"ag_args_ensemble": ag_args_ensemble} if ag_args_ensemble is not None else {}
        return {"model_cls": model_cls, "model_hyperparameters": hyperparameters, "fit_kwargs": fit_kwargs or {}}

    def test_base_default_is_true(self):
        assert AbstractExecModel.uses_ray({}) is True

    def test_no_bag_is_false(self):
        assert AGSingleWrapper.uses_ray(self._kwargs(fit_kwargs={"num_cpus": 4})) is False
        assert AGSingleWrapper.uses_ray(self._kwargs(fit_kwargs={"num_bag_folds": 1})) is False
        assert AGSingleWrapper.uses_ray(self._kwargs(fit_kwargs={"num_bag_folds": None})) is False

    @pytest.mark.parametrize(
        "fit_kwargs",
        [{"fit_strategy": "parallel"}, {"dynamic_stacking": True}, {"dynamic_stacking": "auto"}, {"auto_stack": True}],
    )
    def test_predictor_level_options_are_true(self, fit_kwargs):
        assert AGSingleWrapper.uses_ray(self._kwargs(fit_kwargs=fit_kwargs)) is True

    def test_bag_follows_the_fold_fitting_strategy(self):
        sequential = self._kwargs(
            fit_kwargs={"num_bag_folds": 8, "num_gpus": 0},
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
        )
        parallel = self._kwargs(
            fit_kwargs={"num_bag_folds": 8, "num_gpus": 0}, ag_args_ensemble={"fold_fitting_strategy": "parallel_local"}
        )
        assert AGSingleWrapper.uses_ray(sequential, problem_type="binary") is False
        assert AGSingleWrapper.uses_ray(parallel, problem_type="binary") is True
        assert AGSingleBagWrapper.uses_ray(parallel) is True  # inherited

    def test_unknown_gpu_count_considers_both_variants(self):
        ensemble = {"fold_fitting_strategy": "sequential_local", "fold_fitting_strategy_gpu": "parallel_local"}
        unknown = self._kwargs(fit_kwargs={"num_bag_folds": 8}, ag_args_ensemble=ensemble)
        cpu = self._kwargs(fit_kwargs={"num_bag_folds": 8, "num_gpus": 0}, ag_args_ensemble=ensemble)
        assert AGSingleWrapper.uses_ray(unknown) is True
        assert AGSingleWrapper.uses_ray(cpu) is False

    def test_registry_key_is_resolved(self):
        kwargs = self._kwargs(
            fit_kwargs={"num_bag_folds": 8, "num_gpus": 0},
            ag_args_ensemble={"fold_fitting_strategy": "sequential_local"},
            model_cls="GBM",
        )
        assert AGSingleWrapper.uses_ray(kwargs) is False

    def test_model_wrapper_is_false(self):
        assert AGModelWrapper.uses_ray({"model_cls": _DummyModel, "fit_kwargs": {"num_gpus": 1}}) is False


# --- per-child test artifact: reuse of the timed prediction -----------------------------


class _FakeSingleChildBag:
    """A served bag: one child by default, no post-hoc transform, no chunking; counts forward passes."""

    name = "Dummy_BAG_L1"

    def __init__(self, *, n_children: int = 1, temperature_scalar=None, conformalize=None, max_batch_size=None):
        self.n_children = n_children
        self.temperature_scalar = temperature_scalar
        self.conformalize = conformalize
        self._max_batch_size = max_batch_size
        self.forward_calls = 0
        self._refit_oof = True
        self._oof_fold_val_idx = [np.array([0, 2]), np.array([1, 3])]

    def _get_max_batch_size(self):
        return self._max_batch_size

    def can_predict_proba(self) -> bool:
        return True

    def predict_proba_children(self, X):
        self.forward_calls += 1
        return [np.full(len(X), 0.5, dtype=np.float32) for _ in range(self.n_children)]

    def get_oof_fold_val_idx(self, X, y):
        assert self._refit_oof
        return self._oof_fold_val_idx


class _FakeBagPredictor:
    def __init__(self, learner, model):
        self._learner = learner
        self.model_best = model.name
        self._trainer = SimpleNamespace(load_model=lambda name: model, models={})

    def model_names(self, can_infer: bool = False) -> list[str]:
        return [self.model_best]

    def transform_features(self, data, model):
        return data

    def load_data_internal(self):
        raise AssertionError("the refit branch must not load the internal training data")


def _fake_learner(problem_type: str, label_cleaner):
    """A learner stand-in bound to the real ``_pre_process_predict_proba``."""
    from autogluon.tabular.learner.abstract_learner import AbstractTabularLearner

    learner = SimpleNamespace(
        problem_type=problem_type,
        label_cleaner=label_cleaner,
        class_labels=getattr(label_cleaner, "ordered_class_labels", None),
        class_labels_transformed=getattr(label_cleaner, "ordered_class_labels_transformed", None),
    )
    learner._pre_process_predict_proba = types.MethodType(AbstractTabularLearner._pre_process_predict_proba, learner)
    return learner


def _bag_wrapper(problem_type: str, label_cleaner, model) -> AGSingleBagWrapper:
    metric = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[problem_type]
    wrapper = AGSingleBagWrapper(
        model_cls=_DummyModel,
        model_hyperparameters={},
        problem_type=problem_type,
        eval_metric=get_metric(metric, problem_type=problem_type),
    )
    wrapper.predictor = _FakeBagPredictor(_fake_learner(problem_type, label_cleaner), model)
    return wrapper


_X_TEST = pd.DataFrame({"f0": [0.1, 0.2, 0.3, 0.4]})


class TestPerChildTestFromTimedOutput:
    def test_binary_yields_the_positive_class_column(self):
        cleaner = LabelCleanerBinary(y=pd.Series(["no", "yes", "yes", "no"]), verbose=False)
        negative, positive = cleaner.ordered_class_labels
        p = np.array([0.1, 0.9, 0.6, 0.3])
        proba = pd.DataFrame({negative: 1 - p, positive: p})  # float64 two columns, as the predictor returns
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("binary", cleaner, bag)

        artifact = wrapper.bag_artifact(_X_TEST, y_pred=None, y_pred_proba=proba)

        (child,) = artifact["pred_proba_test_per_child"]
        assert child.dtype == np.float32
        assert np.array_equal(child, p.astype(np.float32))
        assert bag.forward_calls == 0
        assert wrapper.per_child_test_source == "timed_prediction"
        assert [list(v) for v in artifact["val_idx_per_child"]] == [[0, 2], [1, 3]]

    def test_multiclass_drops_the_unseen_class_column(self):
        cleaner = LabelCleanerMulticlass(y=pd.Series(["a", "b", "c", "a"]), y_uncleaned=pd.Series(["a", "b", "c", "d"]))
        assert cleaner.ordered_class_labels == ["a", "b", "c", "d"]
        internal = np.array([[0.7, 0.2, 0.1], [0.1, 0.8, 0.1], [0.2, 0.2, 0.6], [0.5, 0.25, 0.25]], dtype=np.float32)
        proba = pd.DataFrame(internal, columns=["a", "b", "c"])
        proba["d"] = np.float32(0.0)
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("multiclass", cleaner, bag)

        (child,) = wrapper.get_per_child_test(_X_TEST, model=bag, y_pred_proba=proba)

        assert child.dtype == np.float32
        assert np.array_equal(child, internal)
        assert bag.forward_calls == 0

    def test_regression_yields_the_series_values(self):
        cleaner = LabelCleanerDummy(problem_type="regression")
        values = np.array([1.5, 2.5, 3.5, 4.5], dtype=np.float32)
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("regression", cleaner, bag)

        (child,) = wrapper.get_per_child_test(_X_TEST, model=bag, y_pred=pd.Series(values))

        assert child.dtype == np.float32
        assert np.array_equal(child, values)
        assert bag.forward_calls == 0

    @pytest.mark.parametrize(
        "bag_kwargs",
        [{"n_children": 2}, {"temperature_scalar": 1.3}, {"conformalize": 0.1}, {"max_batch_size": 2}],
    )
    def test_fallbacks_take_the_forward_pass(self, bag_kwargs):
        cleaner = LabelCleanerBinary(y=pd.Series(["no", "yes", "yes", "no"]), verbose=False)
        negative, positive = cleaner.ordered_class_labels
        proba = pd.DataFrame({negative: [0.9, 0.1, 0.4, 0.7], positive: [0.1, 0.9, 0.6, 0.3]})
        bag = _FakeSingleChildBag(**bag_kwargs)
        wrapper = _bag_wrapper("binary", cleaner, bag)

        children = wrapper.get_per_child_test(_X_TEST, model=bag, y_pred_proba=proba)

        assert bag.forward_calls == 1
        assert len(children) == bag.n_children
        assert wrapper.per_child_test_source == "child_forward_pass"

    def test_label_cleaner_mismatch_takes_the_forward_pass(self):
        # A multiclass task whose learner collapsed to binary internally: the inverse is not exact.
        y = pd.Series(["a", "b", "a", "b"])
        cleaner = LabelCleanerMulticlassToBinary(y=y, y_uncleaned=y)
        assert cleaner.problem_type_transform == "binary"
        proba = pd.DataFrame({"a": [0.9, 0.1, 0.4, 0.7], "b": [0.1, 0.9, 0.6, 0.3]})
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("multiclass", cleaner, bag)

        wrapper.get_per_child_test(_X_TEST, model=bag, y_pred_proba=proba)

        assert bag.forward_calls == 1
        assert wrapper.per_child_test_source == "child_forward_pass"

    def test_missing_or_misaligned_timed_output_takes_the_forward_pass(self):
        cleaner = LabelCleanerBinary(y=pd.Series(["no", "yes", "yes", "no"]), verbose=False)
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("binary", cleaner, bag)

        wrapper.get_per_child_test(_X_TEST, model=bag)
        assert bag.forward_calls == 1
        assert wrapper._per_child_test_from_timed_output(model=bag, n_rows=4, y_pred=None, y_pred_proba=None) is None

        negative, positive = cleaner.ordered_class_labels
        short = pd.DataFrame({negative: [0.9, 0.1], positive: [0.1, 0.9]})
        assert wrapper._per_child_test_from_timed_output(model=bag, n_rows=4, y_pred=None, y_pred_proba=short) is None


class TestGetPerChildValIdx:
    def test_refit_oof_skips_load_data_internal(self):
        bag = _FakeSingleChildBag()
        wrapper = _bag_wrapper("regression", LabelCleanerDummy(problem_type="regression"), bag)
        val_idx = wrapper.get_per_child_val_idx(model=bag)
        assert [list(v) for v in val_idx] == [[0, 2], [1, 3]]

    def test_bagged_loads_internal_data(self):
        class _SplitterBag(_FakeSingleChildBag):
            def __init__(self):
                super().__init__(n_children=2)
                self._refit_oof = False
                self._oof_fold_val_idx = None

            def get_oof_fold_val_idx(self, X, y):
                assert X is not None and y is not None
                return [np.arange(0, 2), np.arange(2, len(X))]

        class _Predictor(_FakeBagPredictor):
            load_calls = 0

            def load_data_internal(self):
                self.load_calls += 1
                return pd.DataFrame({"f0": [1.0, 2.0, 3.0, 4.0]}), pd.Series([1.0, 2.0, 3.0, 4.0])

        bag = _SplitterBag()
        wrapper = _bag_wrapper("regression", LabelCleanerDummy(problem_type="regression"), bag)
        wrapper.predictor = _Predictor(wrapper.predictor._learner, bag)

        val_idx = wrapper.get_per_child_val_idx(model=bag)

        assert wrapper.predictor.load_calls == 1
        assert [list(v) for v in val_idx] == [[0, 1], [2, 3]]


@pytest.mark.models
class TestBagArtifactReuseNumerics:
    """A real (tiny, CPU-only) RandomForest bag: the reuse path equals the per-child forward pass bit for bit."""

    @pytest.mark.parametrize("problem_type", ["binary", "multiclass", "regression"])
    @pytest.mark.parametrize("refit_folds", [True, False])
    def test_reuse_matches_forward_pass(self, tmp_path, problem_type, refit_folds):
        pytest.importorskip("sklearn")
        from tabarena.utils.synthetic_data import make_synthetic_frames

        X, y, X_test = make_synthetic_frames(problem_type, n_rows=80, n_features=4, n_categorical=1, seed=0)
        metric = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}[problem_type]
        wrapper = AGSingleBagWrapper(
            model_cls="RF",
            model_hyperparameters={
                "n_estimators": 5,
                # use_child_oof=False keeps the bag a real k-fold bag (RF defaults it to True, which would
                # make the single child produce the OOF and skip the refit path).
                "ag_args_ensemble": {
                    "use_child_oof": False,
                    "refit_folds": refit_folds,
                    "fold_fitting_strategy": "sequential_local",
                },
            },
            fit_kwargs={"num_bag_folds": 2, "num_cpus": 1, "num_gpus": 0},
            init_kwargs={"path": str(tmp_path / "ag"), "verbosity": 0},
            problem_type=problem_type,
            eval_metric=get_metric(metric, problem_type=problem_type),
        )
        try:
            out = wrapper.fit_custom(X, y, X_test)
            reused = wrapper.bag_artifact(X_test, y_pred=out["predictions"], y_pred_proba=out["probabilities"])
            reused_source = wrapper.per_child_test_source
            forward = wrapper.bag_artifact(X_test)
        finally:
            wrapper.cleanup()

        assert wrapper.per_child_test_source == "child_forward_pass"
        assert reused_source == ("timed_prediction" if refit_folds else "child_forward_pass")
        n_children = 1 if refit_folds else 2
        assert len(reused["pred_proba_test_per_child"]) == len(forward["pred_proba_test_per_child"]) == n_children
        for a, b in zip(reused["pred_proba_test_per_child"], forward["pred_proba_test_per_child"], strict=True):
            assert a.dtype == b.dtype == np.float32
            assert a.shape == b.shape
            assert np.array_equal(a, b)
        for a, b in zip(reused["val_idx_per_child"], forward["val_idx_per_child"], strict=True):
            assert np.array_equal(a, b)
