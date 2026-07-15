"""Tests for the warm-up integration of the exec models and the experiment runner.

The runner is instantiated via ``object.__new__`` (as in ``test_experiment_runner.py``) to
bypass the heavy ``__init__``; only the attributes ``run_warmup`` / ``_experiment_metadata``
touch are set. Wrapper tests use a dummy AutoGluon model class that records its warm-up.
"""

from __future__ import annotations

from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import AGModelWrapper, AGSingleWrapper
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner


class _RecordingModel(AbstractModel):
    ag_key = "_REC"
    ag_name = "_Recording"
    warmup_calls: list[dict] = []

    @classmethod
    def warmup(cls, **kwargs) -> None:
        cls.warmup_calls.append(kwargs)


def _rmse():
    return get_metric("rmse", problem_type="regression")


# --- exec models -------------------------------------------------------------------------


def test_base_exec_model_has_no_warmup_by_default():
    model = AbstractExecModel(problem_type="regression", eval_metric=_rmse())
    assert model.warmup_fn is None


def test_ag_single_wrapper_warms_its_model_class_with_context():
    _RecordingModel.warmup_calls.clear()
    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={"lr": 0.1},
        fit_kwargs={"num_cpus": 4, "num_gpus": 0},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == [
        {"problem_type": "regression", "num_cpus": 4, "num_gpus": 0, "hyperparameters": {"lr": 0.1}}
    ]


def test_ag_wrapper_skips_unresolvable_model_keys():
    _RecordingModel.warmup_calls.clear()
    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    # A preset-style config with an unknown string key must not break the (best-effort) warm-up.
    wrapper.fit_kwargs["hyperparameters"] = {"NOT_A_REAL_AG_KEY": {}}
    wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == []


def test_ag_model_wrapper_warms_its_model_class_with_context():
    _RecordingModel.warmup_calls.clear()
    wrapper = AGModelWrapper(
        model_cls=_RecordingModel,
        hyperparameters={"lr": 0.2},
        fit_kwargs={"num_gpus": 1},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == [
        {"problem_type": "regression", "num_cpus": None, "num_gpus": 1, "hyperparameters": {"lr": 0.2}}
    ]


# --- experiment runner -------------------------------------------------------------------


class _FakeExecModel:
    def __init__(self, warmup_fn):
        self._warmup_fn = warmup_fn

    @property
    def warmup_fn(self):
        return self._warmup_fn


def _make_runner(*, warmup_fn, warmup: bool = True) -> ExperimentRunner:
    runner = object.__new__(ExperimentRunner)
    runner.warmup = warmup
    runner.method = "MyMethod"
    runner.model = _FakeExecModel(warmup_fn)
    return runner


def test_run_warmup_times_the_warmup():
    calls: list[str] = []
    runner = _make_runner(warmup_fn=lambda: calls.append("warm"))
    duration = runner.run_warmup()
    assert calls == ["warm"]
    assert duration >= 0


def test_run_warmup_nothing_to_warm_returns_none():
    assert _make_runner(warmup_fn=None).run_warmup() is None


def test_run_warmup_disabled_does_not_call():
    calls: list[str] = []
    runner = _make_runner(warmup_fn=lambda: calls.append("warm"), warmup=False)
    assert runner.run_warmup() is None
    assert calls == []


def test_run_warmup_failure_is_non_fatal():
    def boom():
        raise RuntimeError("warm-up bug")

    assert _make_runner(warmup_fn=boom).run_warmup() is None  # logged, fit proceeds cold


def test_experiment_metadata_records_warmup_time():
    runner = object.__new__(ExperimentRunner)
    runner.method_cls = _FakeExecModel
    runner.time_warmup_s = 1.5
    metadata = runner._experiment_metadata(time_start=0.0, time_start_str="")
    assert metadata["time_warmup_s"] == 1.5
