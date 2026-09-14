"""Tests for ``AGSingleWrapper``'s post-fit metadata collection.

A fake predictor/trainer stands in for AutoGluon; only the metadata bookkeeping is under test,
in particular that the best model's ``get_info()`` is collected once and read consistently.
"""

from __future__ import annotations

from types import SimpleNamespace

from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import AGSingleWrapper, _hyperparameters_user_from_info


class _DummyModel(AbstractModel):
    ag_key = "DUMMYMETADATAMODEL"
    ag_name = "DummyMetadataModel"


class _FakeFittedModel:
    """Loaded best model; counts the (expensive) ``get_info`` calls."""

    fit_num_cpus = 4
    fit_num_gpus = 1
    fit_num_cpus_child = 2
    fit_num_gpus_child = 0.5
    _memory_usage_estimate = 12345

    def __init__(self, info: dict):
        self._info = info
        self.get_info_calls = 0

    def get_info(self, include_feature_metadata: bool = True) -> dict:
        assert include_feature_metadata is False
        self.get_info_calls += 1
        return self._info

    def disk_usage(self) -> int:
        return 42

    def get_fit_metadata(self) -> dict:
        return {"fit": "metadata"}


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
