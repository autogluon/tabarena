"""Tests for ``AGSingleWrapper``'s post-fit metadata collection and validation-protocol plumbing.

A fake predictor/trainer stands in for AutoGluon; only the bookkeeping is under test: the best
model's ``get_info()`` is collected once and read consistently (custom-split index arrays stripped),
the fit / init kwargs that would bypass the validation protocol are rejected, and the validation
record reports what the bag itself says.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import (
    AGSingleWrapper,
    AGWrapper,
    _hyperparameters_user_from_info,
    _strip_custom_splits_from_info,
)


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


# --- validation protocol plumbing ---------------------------------------------------------------


class TestStripCustomSplits:
    def test_strips_every_known_location_and_counts(self):
        splits = [(np.arange(3), np.arange(3, 6))]
        info = {
            "hyperparameters_user": {"ag_args_ensemble": {"custom_splits": splits, "num_folds": 8}},
            "hyperparameters": {"custom_splits": splits, "use_child_oof": False},
            "bagged_info": {
                "child_hyperparameters_user": {"ag_args_ensemble": {"custom_splits": splits}},
                "child_hyperparameters": {"custom_splits": splits, "n_estimators": 8},
                "num_child_models": 8,
            },
        }
        stripped, count = _strip_custom_splits_from_info(info)
        assert count == 4
        assert stripped["hyperparameters_user"]["ag_args_ensemble"] == {"num_folds": 8}
        assert stripped["hyperparameters"] == {"use_child_oof": False}
        assert stripped["bagged_info"]["child_hyperparameters_user"]["ag_args_ensemble"] == {}
        assert stripped["bagged_info"]["child_hyperparameters"] == {"n_estimators": 8}
        assert stripped["bagged_info"]["num_child_models"] == 8
        # The input is left untouched.
        assert info["hyperparameters"]["custom_splits"] is splits

    def test_returns_the_same_object_when_nothing_to_strip(self):
        info = {"hyperparameters_user": {"n_estimators": 8}, "bagged_info": {"num_child_models": 8}}
        stripped, count = _strip_custom_splits_from_info(info)
        assert stripped is info
        assert count == 0

    def test_get_metadata_strips_the_collected_info(self):
        # A bag's ``hyperparameters_user`` are its bag-level params, where AutoGluon puts the predictor-level
        # ``custom_splits`` next to the seed.
        splits = [(np.arange(2), np.arange(2, 4))]
        info = {
            "hyperparameters_user": {"custom_splits": splits, "model_random_seed": 3},
            "bagged_info": {"child_hyperparameters_user": {"n_estimators": 8}},
        }
        wrapper = _make_wrapper(_FakeFittedModel(info))
        metadata = wrapper.get_metadata()
        assert metadata["hyperparameters"] == {"n_estimators": 8, "ag_args_ensemble": {"model_random_seed": 3}}
        assert metadata["info"]["hyperparameters_user"] == {"model_random_seed": 3}
        assert info["hyperparameters_user"]["custom_splits"] is splits  # the model's own info is untouched


class TestValidationBypassBans:
    @pytest.mark.parametrize(
        "key",
        ["validation_structure", "use_bag_holdout", "holdout_frac", "tuning_data", "dynamic_stacking", "refit_full"],
    )
    def test_predictor_level_validation_knobs_are_rejected_in_fit_kwargs(self, key):
        with pytest.raises(AssertionError, match="validation protocol"):
            AGSingleWrapper(
                model_cls=_DummyModel,
                model_hyperparameters={},
                fit_kwargs={key: 1},
                problem_type="regression",
                eval_metric=get_metric("rmse", problem_type="regression"),
            )

    @pytest.mark.parametrize("key", ["groups", "learner_kwargs"])
    def test_predictor_init_validation_knobs_are_rejected(self, key):
        with pytest.raises(AssertionError, match="validation protocol"):
            AGSingleWrapper(
                model_cls=_DummyModel,
                model_hyperparameters={},
                init_kwargs={key: 1},
                problem_type="regression",
                eval_metric=get_metric("rmse", problem_type="regression"),
            )

    def test_use_task_specific_validation_is_not_a_wrapper_argument(self):
        with pytest.raises(TypeError, match="task_specific_validation"):
            AGWrapper(problem_type="regression", eval_metric=None, use_task_specific_validation=True)


class TestValidationRecord:
    def test_bag_record_reads_the_collected_info(self):
        info = {
            "hyperparameters_user": {},
            "hyperparameters": {
                "use_child_oof": True,
                "refit_folds": False,
                "fold_fitting_strategy": "sequential_local",
                "n_neighbors": 20,
            },
            "bagged_info": {
                "num_child_models": 1,
                "_n_repeats": 1,
                "_k_per_n_repeat": [8],
                "bagged_mode": True,
                "child_hyperparameters_user": {},
            },
        }
        model = _FakeFittedModel(info)
        model._child_oof = True
        wrapper = _make_wrapper(model)
        wrapper.get_metadata()

        record = wrapper.get_validation_record()
        assert record["num_child_models"] == 1
        assert record["child_oof"] is True
        assert record["k_per_n_repeat"] == [8]
        assert record["bag_params"] == {
            "use_child_oof": True,
            "refit_folds": False,
            "fold_fitting_strategy": "sequential_local",
        }
        # Nothing was resolved in this fake fit and the fake trainer reports no fold counts.
        assert "regime" not in record
        assert record["num_bag_folds_fitted"] is None

    def test_record_before_metadata_is_collected_has_only_the_trainer_part(self):
        wrapper = _make_wrapper(_FakeFittedModel(SINGLE_INFO))
        assert wrapper.get_validation_record() == {"num_bag_folds_fitted": None, "num_bag_sets_fitted": None}
