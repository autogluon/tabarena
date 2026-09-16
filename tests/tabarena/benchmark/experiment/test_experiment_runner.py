"""Tests for ``ExperimentRunner`` / ``OOFExperimentRunner`` outside a real fit.

The runners are instantiated via ``object.__new__`` to bypass the heavy ``__init__`` (which
loads real task data); only the attributes each method touches are set. Covered: the
post-evaluate hooks, the timed predictions reaching ``bag_artifact``, cleanup on failure, the
memoized post-fit reload of a lazy task, the timing audit in the experiment metadata,
``Experiment.uses_ray`` and the shared-weights release between in-process sweep items.
"""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
from autogluon.core.data.label_cleaner import LabelCleanerDummy
from autogluon.core.metrics import get_metric

from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.experiment.experiment_constructor import Experiment
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
    # A model without ``get_validation_record`` contributes an empty fitted part.
    assert out["validation_protocol"] == {}


def test_base_post_evaluate_records_what_the_model_reports_about_its_validation():
    class _RecordingModel(_FakeModel):
        def get_validation_record(self) -> dict:
            return {"regime": "default", "num_bag_folds_fitted": 8}

    runner = _make_runner(ExperimentRunner, problem_type="regression", model=_RecordingModel(), y_test=pd.Series([1.0]))
    out = runner.post_evaluate({})
    assert out["validation_protocol"] == {"regime": "default", "num_bag_folds_fitted": 8}


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


# --- OOF: the timed predictions reach bag_artifact -------------------------------------


class _BagModel(_FakeModel):
    """Exec model exposing per-child artifacts; records the ``bag_artifact`` call."""

    def __init__(self, *, oof: dict, source: str = "timed_prediction"):
        super().__init__(can_get_oof=True, oof=oof)
        self.can_get_per_child_oof = True
        self.can_get_per_child_val_idx = True
        self.per_child_test_source = None
        self._source = source
        self.bag_calls: list[dict] = []

    def get_metadata(self) -> dict:
        return {"per_child_test_source": self.per_child_test_source}

    def bag_artifact(self, X_test, *, y_pred=None, y_pred_proba=None) -> dict:
        self.bag_calls.append({"X_test": X_test, "y_pred": y_pred, "y_pred_proba": y_pred_proba})
        self.per_child_test_source = self._source
        return {"pred_proba_test_per_child": [], "val_idx_per_child": []}


def test_oof_post_evaluate_passes_timed_predictions_to_bag_artifact():
    oof = {"pred_proba_dict_val": pd.Series([1.1, 2.1]), "y_val": pd.Series([1.0, 2.0])}
    preds = pd.Series([1.5, 2.5])
    model = _BagModel(oof=oof)
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="regression",
        model=model,
        y_test=pd.Series([1.0, 2.0]),
        compute_bag_info=True,
    )
    runner.X_test = pd.DataFrame({"a": [0.0, 1.0]})

    out = runner.post_evaluate({"predictions": preds, "probabilities": None})

    (call,) = model.bag_calls
    assert call["X_test"] is runner.X_test
    assert call["y_pred"] is preds  # identity: the runner's own timed output, not a copy
    assert call["y_pred_proba"] is None
    assert out["simulation_artifacts"]["bag_info"] == {"pred_proba_test_per_child": [], "val_idx_per_child": []}
    # get_metadata ran before the artifact; the runner refreshes the source it recorded.
    assert out["method_metadata"]["per_child_test_source"] == "timed_prediction"


def test_oof_post_evaluate_passes_probabilities_by_identity_for_classification():
    oof = {"pred_proba_dict_val": pd.Series([0.1, 0.9]), "y_val": pd.Series([0, 1])}
    probs = pd.DataFrame({0: [0.2, 0.7], 1: [0.8, 0.3]})
    preds = pd.Series([1, 0])
    model = _BagModel(oof=oof, source="child_forward_pass")
    runner = _make_runner(
        OOFExperimentRunner,
        problem_type="binary",
        model=model,
        y_test=pd.Series([0, 1]),
        eval_metric_name="roc_auc",
        compute_bag_info=True,
    )
    runner.X_test = pd.DataFrame({"a": [0.0, 1.0]})

    out = runner.post_evaluate({"predictions": preds, "probabilities": probs})

    (call,) = model.bag_calls
    assert call["y_pred"] is preds
    assert call["y_pred_proba"] is probs
    assert out["method_metadata"]["per_child_test_source"] == "child_forward_pass"


# --- run: cleanup on failure ---------------------------------------------------------------


class _SpyCleanupModel:
    def __init__(self, *, fail: bool = False):
        self.cleanup_calls = 0
        self.fail = fail

    def cleanup(self) -> None:
        self.cleanup_calls += 1
        if self.fail:
            raise RuntimeError("cleanup bug")


def _runner_for_run(*, model, run_raises: bool, cleanup: bool = True, cleanup_on_failure: bool = False):
    runner = object.__new__(ExperimentRunner)
    runner.model = model
    runner.cleanup = cleanup
    runner.cleanup_on_failure = cleanup_on_failure

    def _run():
        if run_raises:
            raise ValueError("fit failed")
        return {"ok": True}

    runner._run = _run
    return runner


def test_run_does_not_clean_up_after_a_failure_by_default():
    model = _SpyCleanupModel()
    runner = _runner_for_run(model=model, run_raises=True)
    with pytest.raises(ValueError, match="fit failed"):
        runner.run()
    assert model.cleanup_calls == 0  # local debugging keeps the artifacts


def test_run_cleans_up_after_a_failure_when_opted_in():
    model = _SpyCleanupModel()
    runner = _runner_for_run(model=model, run_raises=True, cleanup_on_failure=True)
    with pytest.raises(ValueError, match="fit failed"):
        runner.run()
    assert model.cleanup_calls == 1


def test_run_cleanup_on_failure_tolerates_a_missing_model_and_a_failing_cleanup(capsys):
    runner = _runner_for_run(model=None, run_raises=True, cleanup_on_failure=True)
    with pytest.raises(ValueError, match="fit failed"):
        runner.run()  # init_method never assigned a model: no AttributeError from the cleanup path

    model = _SpyCleanupModel(fail=True)
    runner = _runner_for_run(model=model, run_raises=True, cleanup_on_failure=True)
    with pytest.raises(ValueError, match="fit failed"):
        runner.run()  # the original exception propagates, the cleanup error is printed
    assert model.cleanup_calls == 1
    captured = capsys.readouterr()
    assert "Cleanup after the failed run raised" in captured.out
    assert "cleanup bug" in captured.err  # the traceback of the cleanup error


def test_run_success_path_cleans_up_exactly_once():
    model = _SpyCleanupModel()
    runner = _runner_for_run(model=model, run_raises=False, cleanup_on_failure=True)
    assert runner.run() == {"ok": True}
    assert model.cleanup_calls == 1

    model = _SpyCleanupModel()
    runner = _runner_for_run(model=model, run_raises=False, cleanup=False)
    runner.run()
    assert model.cleanup_calls == 0


# --- lazy-loaded test split: one reload after the fit -----------------------------------


def test_lazy_test_split_is_reloaded_once_after_the_fit():
    runner = object.__new__(ExperimentRunner)
    runner.task = SimpleNamespace(lazy_load_data=True)
    runner._post_fit_split = None
    X_test, y_test = pd.DataFrame({"a": [1.0]}), pd.Series([1.0])
    calls: list[int] = []

    def split():
        calls.append(1)
        return pd.DataFrame(), pd.Series(dtype=float), X_test, y_test

    runner._train_test_split = split

    assert runner._load_y_test() is y_test
    assert runner._load_x_test() is X_test
    assert runner._load_y_test() is y_test
    assert len(calls) == 1


def test_run_model_fit_clears_the_memo_and_hands_the_lazy_loader_to_fit_custom():
    runner = object.__new__(ExperimentRunner)
    runner.task = SimpleNamespace(lazy_load_data=True)
    runner.task_split_idx = 7
    runner._post_fit_split = ("stale", "stale")
    received: dict = {}
    runner.model = SimpleNamespace(fit_custom=lambda **kwargs: received.update(kwargs) or {"fit": True})

    assert runner.run_model_fit() == {"fit": True}
    assert runner._post_fit_split is None  # nothing memoized during the timed fit
    assert received["X"] is None and received["lazy_load_function"] == runner._lazy_load_for_run_model_fit
    assert received["split_seed"] == 7


def test_non_lazy_task_reads_the_loaded_split():
    runner = object.__new__(ExperimentRunner)
    runner.task = SimpleNamespace(lazy_load_data=False)
    runner.X_test, runner.y_test = pd.DataFrame({"a": [1.0]}), pd.Series([1.0])
    assert runner._load_x_test() is runner.X_test
    assert runner._load_y_test() is runner.y_test


# --- experiment metadata: the timing audit -----------------------------------------------


def test_experiment_metadata_records_the_timing_audit():
    runner = object.__new__(ExperimentRunner)
    runner.method_cls = _FakeModel
    runner.time_warmup_s = None
    runner.warmup_report = None
    runner.timing_audit = {"fit": {"new_modules": 3}, "predict": None, "scope": "main process"}
    metadata = runner._experiment_metadata(time_start=0.0, time_start_str="")
    assert metadata["timing_audit"] == {"fit": {"new_modules": 3}, "predict": None, "scope": "main process"}

    del runner.timing_audit
    assert runner._experiment_metadata(time_start=0.0, time_start_str="")["timing_audit"] is None


def test_run_moves_the_timing_audit_from_the_fit_output_into_the_experiment_metadata():
    warmup = importlib.import_module("tabarena.models.warmup")
    preds = pd.Series([1.5, 2.5])
    audit = {"fit": {"new_modules": 0}, "predict": {"new_modules": 0}, "scope": "main process"}
    model = _FakeModel(can_get_oof=False)
    runner = _make_runner(ExperimentRunner, problem_type="regression", model=model, y_test=pd.Series([1.0, 2.0]))
    runner.method_cls = _FakeModel
    runner.eval_metric = get_metric("rmse", problem_type="regression")
    runner.debug_mode = True
    runner._post_fit_split = None
    runner.init_method = lambda: model
    runner.run_warmup = lambda: warmup.WarmupReport(status="none")
    runner.run_model_fit = lambda: {
        "predictions": preds,
        "probabilities": None,
        "time_train_s": 1.0,
        "time_infer_s": 0.5,
        "memory_usage": {},
        "timing_audit": audit,
    }

    out = runner._run()

    assert "timing_audit" not in out
    assert out["experiment_metadata"]["timing_audit"] == audit
    assert out["experiment_metadata"]["time_warmup_s"] is None
    assert out["metric_error"] == pytest.approx(0.5)


# --- Experiment.uses_ray ------------------------------------------------------------------


class _RayAwareModel(AbstractExecModel):
    seen: list[tuple] = []

    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        cls.seen.append((method_kwargs, problem_type))
        return False


class _BrokenUsesRayModel(AbstractExecModel):
    @classmethod
    def uses_ray(cls, method_kwargs: dict, *, problem_type: str | None = None) -> bool:
        raise KeyError("model_cls")


def test_experiment_uses_ray_delegates_with_a_copy_of_the_method_kwargs():
    _RayAwareModel.seen.clear()
    experiment = Experiment(name="exp", method_cls=_RayAwareModel, method_kwargs={"fit_kwargs": {"num_bag_folds": 8}})

    assert experiment.uses_ray(problem_type="binary") is False

    (seen_kwargs, seen_problem_type) = _RayAwareModel.seen[-1]
    assert seen_kwargs == experiment.method_kwargs
    assert seen_kwargs is not experiment.method_kwargs  # a deep copy, the experiment is never mutated
    assert seen_problem_type == "binary"


def test_experiment_uses_ray_is_true_on_error():
    experiment = Experiment(name="exp", method_cls=_BrokenUsesRayModel, method_kwargs={})
    assert experiment.uses_ray() is True


# --- in-process sweep: shared-weights release between model classes ------------------------


def test_run_sweep_releases_shared_weights_only_when_the_model_class_changes(monkeypatch):
    api = importlib.import_module("tabarena.benchmark.experiment.experiment_runner_api")
    released: list[int] = []
    monkeypatch.setattr(api, "_release_shared_weights", lambda: released.append(1))

    class _Exp:
        def __init__(self, method_cls, model_cls):
            self.method_cls = method_cls
            self.method_kwargs = {"model_cls": model_cls}

        def run(self, **kwargs) -> dict:
            return {"ok": True}

    def job(experiment, index: int):
        current = (object(), "rmse", "d0")
        return SimpleNamespace(
            model_experiment=experiment,
            lazy_task=SimpleNamespace(current=current, materialize=lambda: current),
            cache_task_key=0,
            fold=0,
            repeat=0,
            cacher=None,
            cache_existed=True,
            input_index=index,
        )

    class _A: ...

    class _B: ...

    jobs = [job(_Exp(_A, "GBM"), 0), job(_Exp(_A, "GBM"), 1), job(_Exp(_A, "CAT"), 2), job(_Exp(_B, "CAT"), 3)]
    results = api._run_sweep(
        jobs, stats=api._RunStats(total=4), ignore_cache=False, debug_mode=False, raise_on_failure=True
    )

    assert released == [1, 1]  # GBM to CAT, then exec model _A to _B; never between the two GBM items
    assert [index for index, _ in results] == [0, 1, 2, 3]
