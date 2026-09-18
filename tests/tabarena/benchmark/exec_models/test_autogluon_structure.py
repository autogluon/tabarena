"""A full-predictor ``AGWrapper`` declares the task's structure to AutoGluon instead of resolving splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.common.utils.validation_structure import ValidationStructure

from tabarena.benchmark.exec_models.autogluon import AGSingleBagWrapper, AGSingleWrapper, AGWrapper
from tabarena.benchmark.exec_models.autogluon_utils import SPLIT_RANDOM_STATE
from tabarena.benchmark.task.metadata import GroupLabelTypes
from tabarena.benchmark.validation_protocol import BEYONDARENA_VALIDATION_PROTOCOL, TABARENA_V0PT1_VALIDATION_PROTOCOL

GROUPED = {"group_on": "grp", "group_labels": GroupLabelTypes.PER_SAMPLE}
TEMPORAL = {"time_on": "t"}
STRATIFIED_ONLY = {"stratify_on": "target"}


def _wrapper(validation_metadata, *, protocol=BEYONDARENA_VALIDATION_PROTOCOL, fit_kwargs=None, init_kwargs=None, **kw):
    return AGWrapper(
        problem_type="regression",
        eval_metric=None,
        validation_metadata=validation_metadata,
        validation_protocol=protocol,
        fit_kwargs=fit_kwargs,
        init_kwargs=init_kwargs,
        **kw,
    )


def _data(n: int = 60):
    X = pd.DataFrame({"feature": np.arange(n, dtype=float), "grp": [f"g{i % 6}" for i in range(n)], "t": np.arange(n)})
    return X, pd.Series(np.zeros(n))


class TestDeclaredStructure:
    def test_grouped_bagged_declares_the_structure_and_keeps_the_counts(self):
        wrapper = _wrapper(GROUPED, fit_kwargs={"num_bag_folds": 4, "num_bag_sets": 2})
        X, y = _data()
        _, init_kwargs, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        structure = fit_kwargs["validation_structure"]
        assert isinstance(structure, ValidationStructure)
        assert structure.group_on == "grp"
        assert structure.time_on is None
        assert fit_kwargs["num_bag_folds"] == 4
        assert fit_kwargs["num_bag_sets"] == 2
        assert "custom_splits" not in fit_kwargs.get("ag_args_ensemble", {})
        assert "tuning_data" not in fit_kwargs
        assert init_kwargs["learner_kwargs"]["random_state"] == SPLIT_RANDOM_STATE
        record = wrapper.get_validation_record()
        assert record["validation_structure"] is True
        assert record["custom_splits"] is False
        assert record["regime"] == "explicit"
        assert record["num_bag_folds_resolved"] == 4
        assert record["structure"]["group_on"] == "grp"

    def test_grouped_holdout_is_left_to_autogluon(self):
        """No counts: a holdout fit. The wrapper carves nothing and declares the structure for AutoGluon's holdout."""
        wrapper = _wrapper(GROUPED)
        X, y = _data()
        train_data, init_kwargs, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert len(train_data) == len(X)
        assert "tuning_data" not in fit_kwargs
        assert fit_kwargs["validation_structure"].group_on == "grp"
        assert init_kwargs["learner_kwargs"]["random_state"] == SPLIT_RANDOM_STATE
        record = wrapper.get_validation_record()
        assert record["validation_structure"] is True
        assert record["task_specific_holdout"] is False
        assert record["regime"] is None
        assert record["num_bag_folds_resolved"] is None

    def test_temporal_forward_only_reaches_the_structure(self):
        wrapper = _wrapper(TEMPORAL, fit_kwargs={"num_bag_folds": 3}, temporal_forward_only=True)
        X, y = _data()
        _, _, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        structure = fit_kwargs["validation_structure"]
        assert structure.time_on == "t"
        assert structure.temporal_forward_only is True

    def test_explicit_learner_seed_wins(self):
        wrapper = _wrapper(
            GROUPED, fit_kwargs={"num_bag_folds": 4}, init_kwargs={"learner_kwargs": {"random_state": 7}}
        )
        X, y = _data()
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert init_kwargs["learner_kwargs"]["random_state"] == 7

    def test_split_random_state_is_configurable(self):
        wrapper = _wrapper(GROUPED, fit_kwargs={"num_bag_folds": 4}, split_random_state=11)
        X, y = _data()
        _, init_kwargs, _ = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert init_kwargs["learner_kwargs"]["random_state"] == 11

    @pytest.mark.parametrize("metadata", [STRATIFIED_ONLY, None])
    def test_unstructured_task_keeps_the_default_splitter_and_seed(self, metadata):
        wrapper = _wrapper(metadata, fit_kwargs={"num_bag_folds": 4})
        X, y = _data()
        _, init_kwargs, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert "validation_structure" not in fit_kwargs
        assert "learner_kwargs" not in init_kwargs
        assert fit_kwargs["num_bag_folds"] == 4
        assert wrapper.get_validation_record()["validation_structure"] is False

    def test_protocol_without_task_specific_validation_declares_nothing(self):
        wrapper = _wrapper(GROUPED, protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, fit_kwargs={"num_bag_folds": 8})
        X, y = _data()
        _, init_kwargs, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert "validation_structure" not in fit_kwargs
        assert "learner_kwargs" not in init_kwargs
        assert wrapper.get_validation_record()["validation_structure"] is False

    def test_no_protocol_declares_nothing(self):
        wrapper = _wrapper(GROUPED, protocol=None, fit_kwargs={"num_bag_folds": 8})
        X, y = _data()
        _, _, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert "validation_structure" not in fit_kwargs
        assert fit_kwargs["num_bag_folds"] == 8

    def test_class_adaptive_folds_follow_the_protocol(self):
        protocol = BEYONDARENA_VALIDATION_PROTOCOL
        wrapper = _wrapper(GROUPED, protocol=protocol, fit_kwargs={"num_bag_folds": 4})
        X, y = _data()
        _, _, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert fit_kwargs.get("adapt_num_bag_folds_to_n_classes", False) is protocol.adapt_num_folds_to_n_classes


class TestSingleModelWrappersKeepResolving:
    def test_single_bag_wrapper_still_produces_custom_splits(self):
        wrapper = AGSingleBagWrapper(
            model_cls="GBM",
            model_hyperparameters={},
            problem_type="regression",
            eval_metric=None,
            validation_metadata=GROUPED,
            validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL,
        )
        X, y = _data(600)
        X["grp"] = [f"g{i % 30}" for i in range(600)]
        _, init_kwargs, fit_kwargs = wrapper._build_predictor_args(X=X, y=y, X_val=None, y_val=None)
        assert "validation_structure" not in fit_kwargs
        assert "learner_kwargs" not in init_kwargs
        assert fit_kwargs["ag_args_ensemble"]["custom_splits"] is not None
        assert wrapper.get_validation_record()["validation_structure"] is False

    @pytest.mark.parametrize("kwargs", [{"temporal_forward_only": True}, {"split_random_state": 1}])
    def test_full_predictor_knobs_are_rejected_on_single_wrappers(self, kwargs):
        with pytest.raises(ValueError, match="full predictor"):
            AGSingleWrapper(
                model_cls="GBM", model_hyperparameters={}, problem_type="regression", eval_metric=None, **kwargs
            )
