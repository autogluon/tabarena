"""Tests for the warm-up integration of the exec models and the experiment runner.

The runner is instantiated via ``object.__new__`` (as in ``test_experiment_runner.py``) to
bypass the heavy ``__init__``; only the attributes ``run_warmup`` / ``_experiment_metadata``
touch are set. Wrapper tests use a dummy AutoGluon model class that records its warm-up; the
dummy fit is turned off on it because it has no ``_fit``.
"""

from __future__ import annotations

import importlib

import pytest
from autogluon.core.metrics import get_metric
from autogluon.core.models import AbstractModel

from tabarena.benchmark.exec_models.autogluon import AGModelWrapper, AGSingleWrapper, AGWrapper
from tabarena.benchmark.exec_models.base import AbstractExecModel
from tabarena.benchmark.exec_models.external import ExternalSystemModel
from tabarena.benchmark.experiment.experiment_runner import ExperimentRunner
from tabarena.utils import ray_utils


def _wu():
    """The live ``tabarena.models.warmup`` module.

    Resolved by name on every use because ``tests/tabarena/models/test_lazy_imports.py`` purges
    ``tabarena.models.*`` from ``sys.modules``; the exec models import the module by name at call
    time, so monkeypatches and ``WarmupReport`` instances must come from the same module object.
    """
    return importlib.import_module("tabarena.models.warmup")


def _is_report(obj) -> bool:
    return isinstance(obj, _wu().WarmupReport)


class _RecordingModel(AbstractModel):
    ag_key = "_REC"
    ag_name = "_Recording"
    warmup_dummy_fit = False
    warmup_calls: list[dict] = []

    @classmethod
    def warmup(cls, **kwargs) -> None:
        cls.warmup_calls.append(kwargs)


def _rmse():
    return get_metric("rmse", problem_type="regression")


@pytest.fixture
def ag_stack_calls(monkeypatch):
    calls: list[dict] = []
    monkeypatch.setattr(_wu(), "warmup_ag_stack", lambda **kw: calls.append(kw))
    return calls


@pytest.fixture
def recorded_imports(monkeypatch):
    imported: list[str] = []

    def fake(*names, report=None):
        imported.extend(names)
        return list(names)

    monkeypatch.setattr(_wu(), "warmup_imports_best_effort", fake)
    return imported


# --- exec models -------------------------------------------------------------------------


def test_base_exec_model_has_no_warmup_by_default():
    model = AbstractExecModel(problem_type="regression", eval_metric=_rmse())
    assert model.warmup_fn is None
    assert AbstractExecModel.warmup_dummy_fit is True
    assert ExternalSystemModel.warmup_dummy_fit is False


def test_base_exec_model_declarative_warmup(recorded_imports):
    class _Declaring(AbstractExecModel):
        warmup_modules = ("wave",)

    model = _Declaring(problem_type="regression", eval_metric=_rmse())
    assert model.warmup_fn is not None
    report = model.warmup_fn()
    assert _is_report(report)
    assert recorded_imports == ["wave"]


def test_external_system_declarative_cuda_from_num_gpus(monkeypatch):
    calls: list[dict] = []

    def fake(**kw):
        calls.append(kw)
        return ["torch:import"]

    monkeypatch.setattr(_wu(), "warmup_torch", fake)

    class _System(ExternalSystemModel):
        warmup_torch_device = True

    system = _System(fit_kwargs={"num_gpus": 0}, problem_type="regression", eval_metric=_rmse())
    report = system.warmup_fn()
    assert calls == [{"cuda": False}]
    assert report.steps == ["torch:import"]
    assert _System(problem_type="regression", eval_metric=_rmse()).warmup_fn() is not None
    assert calls[-1] == {"cuda": None}


def test_ag_single_wrapper_warms_its_model_class_with_context(ag_stack_calls):
    _RecordingModel.warmup_calls.clear()
    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={"lr": 0.1},
        fit_kwargs={"num_cpus": 4, "num_gpus": 0},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    report = wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == [
        {"problem_type": "regression", "num_cpus": 4, "num_gpus": 0, "hyperparameters": {"lr": 0.1}}
    ]
    assert _is_report(report)
    assert report.model_classes == ["_RecordingModel"]
    assert len(ag_stack_calls) == 1 and ag_stack_calls[0]["report"] is report
    assert report.ray["skipped"] == ray_utils.DISABLE_RAY_WARMUP_ENV  # tests/conftest.py kill switch


def test_ag_wrapper_skips_unresolvable_model_keys(ag_stack_calls):
    _RecordingModel.warmup_calls.clear()
    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    # A preset-style config with an unknown string key must not break the (best-effort) warm-up.
    wrapper.fit_kwargs["hyperparameters"] = {"NOT_A_REAL_AG_KEY": {}}
    report = wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == []
    assert report.model_classes == []


def test_ag_wrapper_declared_modules_apply(ag_stack_calls, recorded_imports):
    class _Declaring(AGSingleWrapper):
        warmup_modules = ("wave",)

    wrapper = _Declaring(
        model_cls=_RecordingModel, model_hyperparameters={}, problem_type="regression", eval_metric=_rmse()
    )
    wrapper.warmup_fn()
    assert recorded_imports[0] == "wave"


def test_ag_wrapper_dummy_fit_flag_is_passed_down(ag_stack_calls, monkeypatch):
    seen: list[bool] = []

    def fake_dummy_fit(model_cls, **kwargs):
        seen.append(True)
        return {}

    monkeypatch.setattr(_wu(), "warmup_dummy_fit", fake_dummy_fit)

    class _NoDummy(AGSingleWrapper):
        warmup_dummy_fit = False

    class _Plain(AbstractModel):
        ag_key = "_PLAINREC"
        ag_name = "_PlainRec"

    wrapper = _NoDummy(model_cls=_Plain, model_hyperparameters={}, problem_type="regression", eval_metric=_rmse())
    report = wrapper.warmup_fn()
    assert seen == [] and "dummy_fit:_Plain:skipped" in report.steps
    wrapper = AGSingleWrapper(
        model_cls=_Plain, model_hyperparameters={}, problem_type="regression", eval_metric=_rmse()
    )
    wrapper.warmup_fn()
    assert seen == [True]


class _RayOk:
    @staticmethod
    def is_initialized():
        return False


def test_ag_wrapper_requests_ray_only_for_cpu_parallel_bags(ag_stack_calls, monkeypatch):
    monkeypatch.delenv(ray_utils.DISABLE_RAY_WARMUP_ENV, raising=False)
    monkeypatch.delenv(ray_utils.RAY_WORKER_WARMUP_ENV, raising=False)
    monkeypatch.setattr(ray_utils, "try_import_ray", lambda: _RayOk())
    init_calls: list[dict] = []

    def fake_init(*, num_cpus, num_gpus):
        init_calls.append({"num_cpus": num_cpus, "num_gpus": num_gpus})
        return {"initialized_by_warmup": True}

    monkeypatch.setattr(ray_utils, "ensure_ray_initialized", fake_init)
    monkeypatch.setattr(
        ray_utils, "warmup_ray_workers", lambda *a, **k: (_ for _ in ()).throw(AssertionError("pool is opt-in"))
    )

    def make(model_hyperparameters=None, **fit_kwargs):
        return AGSingleWrapper(
            model_cls=_RecordingModel,
            model_hyperparameters=model_hyperparameters or {},
            fit_kwargs=fit_kwargs,
            problem_type="regression",
            eval_metric=_rmse(),
        )

    report = make(num_bag_folds=8, num_cpus=8, num_gpus=0).warmup_fn()
    assert init_calls == [{"num_cpus": 8, "num_gpus": 0}]
    assert report.ray["initialized_by_warmup"] is True and "ray:init" in report.steps
    assert report.ray["fold_fitting_strategies"] == ["parallel_local"]

    init_calls.clear()
    hps = {"ag_args_ensemble": {"fold_fitting_strategy": "sequential_local"}}
    report = make(model_hyperparameters=hps, num_bag_folds=8, num_cpus=8, num_gpus=0).warmup_fn()
    assert init_calls == [] and report.ray["skipped"] == "no configured class uses parallel fold fitting"

    report = make(num_bag_folds=8, num_cpus=8, num_gpus=1).warmup_fn()
    assert init_calls == [] and report.ray["skipped"] == "GPU bag; AutoGluon starts Ray itself"

    report = make(num_cpus=8, num_gpus=0).warmup_fn()
    assert init_calls == [] and report.ray["skipped"] == "not a bagged fit"

    report = make(num_bag_folds=8, num_cpus=8).warmup_fn()
    assert init_calls == [] and report.ray["skipped"] == "num_gpus unknown"


def test_ag_wrapper_starts_worker_pool_only_when_opted_in(ag_stack_calls, monkeypatch):
    monkeypatch.delenv(ray_utils.DISABLE_RAY_WARMUP_ENV, raising=False)
    monkeypatch.setenv(ray_utils.RAY_WORKER_WARMUP_ENV, "1")
    monkeypatch.setattr(ray_utils, "try_import_ray", lambda: _RayOk())
    monkeypatch.setattr(ray_utils, "ensure_ray_initialized", lambda **kw: {"initialized_by_warmup": True})
    pool_calls: list[dict] = []

    def fake_pool(modules, num_workers, *, cpus_per_worker):
        pool_calls.append({"modules": list(modules), "num_workers": num_workers, "cpus_per_worker": cpus_per_worker})
        return {"requested": num_workers}

    monkeypatch.setattr(ray_utils, "warmup_ray_workers", fake_pool)
    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={},
        fit_kwargs={"num_bag_folds": 8, "num_cpus": 16, "num_gpus": 0},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    report = wrapper.warmup_fn()
    assert pool_calls == [
        {"modules": list(_wu().ray_worker_warmup_modules(_RecordingModel)), "num_workers": 8, "cpus_per_worker": 2}
    ]
    assert report.ray["pool"] == {"requested": 8} and "ray:pool" in report.steps


def test_ag_wrapper_warms_feature_generator_from_fit_kwargs(ag_stack_calls, recorded_imports):
    class _Gen:
        warmup_modules = ("wave",)

    wrapper = AGSingleWrapper(
        model_cls=_RecordingModel,
        model_hyperparameters={},
        fit_kwargs={"feature_generator_cls": _Gen, "feature_generator_kwargs": {}},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    wrapper.warmup_fn()
    assert "wave" in recorded_imports


def test_ag_wrapper_resolves_presets_for_warmup():
    from autogluon.tabular.models import CatBoostModel, LGBModel

    wrapper = AGWrapper(fit_kwargs={"presets": "extreme_quality"}, problem_type="regression", eval_metric=_rmse())
    classes = [cls for cls, _ in wrapper._configured_model_classes()]
    assert LGBModel in classes and CatBoostModel in classes
    assert wrapper.fit_kwargs == {"presets": "extreme_quality"}  # resolution never edits the fit kwargs


def test_ag_model_wrapper_warms_its_model_class_with_context(recorded_imports):
    _RecordingModel.warmup_calls.clear()
    wrapper = AGModelWrapper(
        model_cls=_RecordingModel,
        hyperparameters={"lr": 0.2},
        fit_kwargs={"num_gpus": 1},
        problem_type="regression",
        eval_metric=_rmse(),
    )
    report = wrapper.warmup_fn()
    assert _RecordingModel.warmup_calls == [
        {"problem_type": "regression", "num_cpus": None, "num_gpus": 1, "hyperparameters": {"lr": 0.2}}
    ]
    assert _is_report(report) and report.model_classes == ["_RecordingModel"]
    assert report.ray == {}  # no AutoGluon stack or Ray on this path


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


def _boom():
    raise RuntimeError("warm-up bug")


def _partial():
    report = _wu().WarmupReport()
    report.step("import:x", failed=True, error=ImportError("x"))
    return report


def test_check_warmup_report_requires_success_by_default():
    runner = _make_runner(warmup_fn=_boom)
    with pytest.raises(RuntimeError, match="warm-up bug"):
        runner.check_warmup_report(runner.run_warmup())
    runner = _make_runner(warmup_fn=_partial)
    with pytest.raises(RuntimeError, match="import:x"):
        runner.check_warmup_report(runner.run_warmup())
    # Nothing to warm and an explicitly disabled warm-up are not failures.
    runner = _make_runner(warmup_fn=None)
    runner.check_warmup_report(runner.run_warmup())
    runner = _make_runner(warmup_fn=_boom, warmup=False)
    runner.check_warmup_report(runner.run_warmup())


def test_check_warmup_report_bypass_fits_cold():
    runner = _make_runner(warmup_fn=_boom)
    runner.require_warmup = False
    runner.check_warmup_report(runner.run_warmup())


def test_run_warmup_times_the_warmup():
    calls: list[str] = []
    runner = _make_runner(warmup_fn=lambda: calls.append("warm"))
    report = runner.run_warmup()
    assert calls == ["warm"]
    assert report.status == "ok" and report.duration_s >= 0
    assert report.label == "MyMethod"


def test_run_warmup_nothing_to_warm():
    report = _make_runner(warmup_fn=None).run_warmup()
    assert report.status == "none" and report.duration_s is None


def test_run_warmup_disabled_does_not_call():
    calls: list[str] = []
    runner = _make_runner(warmup_fn=lambda: calls.append("warm"), warmup=False)
    report = runner.run_warmup()
    assert report.status == "disabled" and report.duration_s is None
    assert calls == []


def test_run_warmup_failure_is_non_fatal(capsys):
    def boom():
        raise RuntimeError("warm-up bug")

    report = _make_runner(warmup_fn=boom).run_warmup()  # logged, fit proceeds cold
    assert report.status == "failed" and report.duration_s is None
    assert "warm-up bug" in report.error
    assert "Warm-up of method 'MyMethod' failed" in capsys.readouterr().out


def test_run_warmup_partial_keeps_duration():
    def partial():
        report = _wu().WarmupReport()
        report.step("import:x", failed=True, error=ImportError("x"))
        return report

    report = _make_runner(warmup_fn=partial).run_warmup()
    assert report.status == "partial" and report.duration_s is not None


def test_experiment_metadata_records_warmup_time_and_report():
    runner = object.__new__(ExperimentRunner)
    runner.method_cls = _FakeExecModel
    runner.warmup_report = _wu().WarmupReport(duration_s=1.5, model_classes=["X"])
    runner.time_warmup_s = runner.warmup_report.duration_s
    metadata = runner._experiment_metadata(time_start=0.0, time_start_str="")
    assert metadata["time_warmup_s"] == 1.5
    assert metadata["warmup_report"]["duration_s"] == 1.5
    assert metadata["warmup_report"]["model_classes"] == ["X"]

    runner.warmup_report = None
    runner.time_warmup_s = None
    metadata = runner._experiment_metadata(time_start=0.0, time_start_str="")
    assert metadata["time_warmup_s"] is None and metadata["warmup_report"] is None
