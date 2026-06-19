"""Tests for ``ExperimentRunner.post_evaluate`` / ``OOFExperimentRunner.post_evaluate``.

The runners are instantiated via ``object.__new__`` to bypass the heavy ``__init__`` (which
loads real task data); only the attributes ``post_evaluate`` touches are set.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from autogluon.core.data.label_cleaner import LabelCleanerDummy

from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner, OOFExperimentRunner


class _FakeTask:
    def __init__(self, problem_type: str, *, label: str = "target"):
        self.problem_type = problem_type
        self.label = label
        self.lazy_load_data = False
        self.task_id = 42


class _FakeModel:
    """Minimal stand-in for an exec model (no ``get_metadata`` -> base skips it)."""

    def __init__(self, *, can_get_oof: bool = True, oof: dict | None = None):
        self.can_get_oof = can_get_oof
        self.can_get_error_val = False
        self.can_get_per_child_oof = False
        self.can_get_per_child_val_idx = False
        self._oof = oof or {}

    def get_oof(self) -> dict:
        return dict(self._oof)  # fresh copy each call, like the real artifact


def _make_runner(
    runner_cls,
    *,
    problem_type: str,
    model: _FakeModel,
    y_test: pd.Series,
    label: str = "target",
    method: str = "MyMethod",
    eval_metric_name: str = "rmse",
    **oof_flags,
):
    runner = object.__new__(runner_cls)
    runner.task = _FakeTask(problem_type, label=label)
    runner.task_name = "d0"
    runner.fold = 0
    runner.repeat = 0
    runner.sample = 0
    runner.task_split_idx = 0
    runner.method = method
    runner.eval_metric_name = eval_metric_name
    runner.model = model
    runner.label_cleaner = LabelCleanerDummy(problem_type=problem_type)
    runner.y_test = y_test
    if runner_cls is OOFExperimentRunner:
        runner.compute_simulation_artifacts = oof_flags.get("compute_simulation_artifacts", True)
        runner.compute_bag_info = oof_flags.get("compute_bag_info", False)
        runner.optimize_simulation_artifacts_memory = oof_flags.get("optimize_simulation_artifacts_memory", False)
    return runner


# --- base ExperimentRunner --------------------------------------------------------------


def test_base_post_evaluate_sets_simulation_artifacts_none_and_metadata():
    runner = _make_runner(ExperimentRunner, problem_type="regression", model=_FakeModel(), y_test=pd.Series([1.0]))
    out = runner.post_evaluate({})
    # The OOF runner's early-return relies on the base setting this to None.
    assert out["simulation_artifacts"] is None
    assert out["framework"] == "MyMethod"
    assert out["problem_type"] == "regression"
    assert out["metric"] == "rmse"
    assert out["task_metadata"] == {"tid": 42, "name": "d0", "fold": 0, "repeat": 0, "sample": 0, "split_idx": 0}


# --- OOF: artifact NOT built -> None ----------------------------------------------------


def test_oof_post_evaluate_disabled_leaves_artifact_none():
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="regression",
        model=_FakeModel(can_get_oof=True),
        y_test=pd.Series([1.0, 2.0]),
        compute_simulation_artifacts=False,
    )
    out = runner.post_evaluate({"predictions": pd.Series([1.0, 2.0]), "probabilities": None})
    assert out["simulation_artifacts"] is None


def test_oof_post_evaluate_model_cannot_get_oof_leaves_artifact_none():
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="regression",
        model=_FakeModel(can_get_oof=False),
        y_test=pd.Series([1.0, 2.0]),
    )
    out = runner.post_evaluate({"predictions": pd.Series([1.0, 2.0]), "probabilities": None})
    assert out["simulation_artifacts"] is None


# --- OOF: artifact built ----------------------------------------------------------------


def test_oof_post_evaluate_builds_artifact_regression():
    oof = {"pred_proba_dict_val": pd.Series([1.1, 2.1, 3.1]), "y_val": pd.Series([1.0, 2.0, 3.0])}
    preds = pd.Series([1.5, 2.5])
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="regression",
        model=_FakeModel(can_get_oof=True, oof=oof),
        y_test=pd.Series([1.0, 2.0]),
    )
    out = runner.post_evaluate({"predictions": preds, "probabilities": None})
    art = out["simulation_artifacts"]
    assert art is not None
    assert art["label"] == "target"
    assert art["metric"] == "rmse"
    # val/test predictions are wrapped under the method name
    assert set(art["pred_proba_dict_val"]) == {"MyMethod"}
    assert set(art["pred_proba_dict_test"]) == {"MyMethod"}
    # regression test "proba" == point predictions (identity transform via dummy cleaner)
    pd.testing.assert_series_equal(art["pred_proba_dict_test"]["MyMethod"], preds)
    pd.testing.assert_series_equal(art["y_test"], pd.Series([1.0, 2.0]))


def test_oof_post_evaluate_builds_artifact_binary_uses_positive_class_column():
    oof = {"pred_proba_dict_val": pd.Series([0.1, 0.9]), "y_val": pd.Series([0, 1])}
    probs = pd.DataFrame({0: [0.2, 0.7], 1: [0.8, 0.3]})
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="binary",
        model=_FakeModel(can_get_oof=True, oof=oof),
        y_test=pd.Series([0, 1]),
        eval_metric_name="roc_auc",
    )
    out = runner.post_evaluate({"predictions": pd.Series([1, 0]), "probabilities": probs})
    art = out["simulation_artifacts"]
    assert art is not None
    # binary stores only the positive-class column, wrapped under the method name
    assert list(art["pred_proba_dict_test"]["MyMethod"]) == [0.8, 0.3]


def test_oof_post_evaluate_optimizes_artifact_memory():
    oof = {
        "pred_proba_dict_val": pd.Series([1.1, 2.1], index=[10, 20]),
        "y_val": pd.Series([1.0, 2.0], index=[10, 20]),
    }
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="regression",
        model=_FakeModel(can_get_oof=True, oof=oof),
        y_test=pd.Series([3.0, 4.0], index=[0, 1]),
        optimize_simulation_artifacts_memory=True,
    )
    out = runner.post_evaluate({"predictions": pd.Series([3.1, 4.1], index=[0, 1]), "probabilities": None})
    art = out["simulation_artifacts"]
    # pandas y/probas replaced by raw numpy arrays, indices stored separately, probas float32
    assert isinstance(art["y_test"], np.ndarray)
    assert isinstance(art["y_val"], np.ndarray)
    assert "y_test_idx" in art
    assert "y_val_idx" in art
    assert art["pred_proba_dict_test"]["MyMethod"].dtype == np.float32
    assert art["pred_proba_dict_val"]["MyMethod"].dtype == np.float32
