"""Tests for the layered, additive warm-up dispatch and its helpers.

Heavy libraries are never warmed for real: ``warmup_torch`` and the best-effort imports are
monkeypatched where they would matter, so these exercise the dispatch order (declared ``warmup``
classmethod, torch layer, ``warmup_modules`` over the MRO, ``WARMUP_STEPS_BY_AG_KEY``, shared
weights, dummy fit), the per-step failure isolation, the CUDA gating and the report contents. The
dummy-fit tests use a tiny ``AbstractModel`` that stores the label mean and stay on the CPU. torch is
optional for them (an install without the benchmark extra has none): the model draws from torch's
generator only when torch is importable, and the tests that check the torch RNG and thread
restoration skip without it. CI installs the CPU build of torch, so all of them run there.
"""

from __future__ import annotations

import logging
import random
import sys

import numpy as np
import pytest
from autogluon.core.models import AbstractModel
from autogluon.tabular.models.abstract.abstract_torch_model import AbstractTorchModel

import tabarena.models.warmup as wu
from tabarena.utils import ray_utils

# --- fakes ------------------------------------------------------------------------------------


class _DeclaredWarmupModel:
    """Model class opting in via the ``warmup`` classmethod convention."""

    warmup_dummy_fit = False
    calls: list[dict] = []

    @classmethod
    def warmup(cls, **kwargs) -> None:
        cls.calls.append(kwargs)


class _TorchModel(AbstractTorchModel):
    warmup_dummy_fit = False


class _MappedModel:
    ag_key = "GBM"
    warmup_dummy_fit = False


class _PlainModel:
    ag_key = "SOME_UNKNOWN_KEY"
    warmup_dummy_fit = False


class _TorchWithModules(AbstractTorchModel):
    warmup_dummy_fit = False
    warmup_modules = ("wave", "this_module_does_not_exist_xyz")


class _ChildWithModules(_TorchWithModules):
    warmup_modules = ("colorsys",)


class _DeclaredTorchModel(AbstractTorchModel):
    warmup_dummy_fit = False
    order: list[str] = []

    @classmethod
    def warmup(cls, **kwargs) -> None:
        cls.order.append("classmethod")


class _RaisingWarmupModel(AbstractTorchModel):
    warmup_dummy_fit = False

    @classmethod
    def warmup(cls, **kwargs) -> None:
        raise RuntimeError("classmethod bug")


class _TabICLLike(AbstractTorchModel):
    ag_key = "TABICL"
    warmup_dummy_fit = False


class _EBMLike:
    ag_key = "EBM"
    warmup_dummy_fit = False


def _torch_or_none():
    """The torch module when it is installed, else ``None`` (an install without the benchmark extra)."""
    try:
        import torch
    except ImportError:
        return None
    return torch


class _MeanModel(AbstractModel):
    """Stores the label mean; predicts constants. Draws from every global RNG to test the guard."""

    ag_key = "_WARMUP_MEAN"
    ag_name = "_WarmupMean"
    fits: list[dict] = []

    def _fit(self, X, y, num_cpus=1, num_gpus=0, time_limit=None, **kwargs):
        X = self.preprocess(X, is_train=True)
        np.random.random()
        random.random()  # noqa: S311
        torch = _torch_or_none()
        if torch is not None:
            torch.rand(1)
        self._mean = float(y.mean())
        type(self).fits.append(
            {
                "n_rows": len(X),
                "n_cols": X.shape[1],
                "dtypes": [str(dt) for dt in X.dtypes],
                "num_cpus": num_cpus,
                "num_gpus": num_gpus,
                "time_limit": time_limit,
                "params": dict(self.params),
                "path": self.path,
            }
        )

    def _predict_proba(self, X, **kwargs):
        X = self.preprocess(X)
        n = len(X)
        if self.problem_type == "regression":
            return np.full(n, self._mean)
        if self.problem_type == "binary":
            return np.full(n, 0.5)
        return np.full((n, 3), 1 / 3)


class _CheapMeanModel(_MeanModel):
    ag_key = "_WARMUP_MEAN_CHEAP"
    warmup_dummy_fit_kwargs = {"n_rows": 40, "n_features": 4, "n_categorical": 2, "time_limit": 7}
    cheap_hyperparameters = {"n_estimators": 1}


class _GpuOnlyMeanModel(_MeanModel):
    ag_key = "_WARMUP_MEAN_GPU"
    minimum_num_gpus = 1
    gpu_required = True


class _GpuPreferringMeanModel(_MeanModel):
    ag_key = "_WARMUP_MEAN_GPU_PREF"
    minimum_num_gpus = 0.5


class _RegressionOnlyMeanModel(_MeanModel):
    ag_key = "_WARMUP_MEAN_REG"
    _supported_problem_types = ["regression"]


class _OptedOutMeanModel(_MeanModel):
    ag_key = "_WARMUP_MEAN_OFF"
    warmup_dummy_fit = False


class _BoomModel(AbstractModel):
    ag_key = "_WARMUP_BOOM"
    ag_name = "_WarmupBoom"

    def _fit(self, X, y, **kwargs):
        raise RuntimeError("fit exploded")


@pytest.fixture
def no_torch_warmup(monkeypatch):
    calls: list[dict] = []

    def fake(**kw):
        calls.append(kw)
        return ["torch:import"]

    monkeypatch.setattr(wu, "warmup_torch", fake)
    return calls


@pytest.fixture
def recorded_imports(monkeypatch):
    imported: list[str] = []

    def fake(*names, report=None):
        imported.extend(names)
        if report is not None:
            for name in names:
                report.step(f"import:{name}")
        return list(names)

    monkeypatch.setattr(wu, "warmup_imports_best_effort", fake)
    return imported


@pytest.fixture
def throwaway_package(tmp_path, monkeypatch):
    """A real, non-stdlib importable package (the imported-modules audit filters the stdlib)."""
    name = "tawarm_probe_pkg"
    (tmp_path / name).mkdir()
    (tmp_path / name / "__init__.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop(name, None)
    yield name
    sys.modules.pop(name, None)


@pytest.fixture(autouse=True)
def fresh_warm_memo():
    """Every test starts in a process that has warmed nothing (the memo is process-wide)."""
    wu.reset_warm_memo()
    yield
    wu.reset_warm_memo()


# --- imports and steps ---------------------------------------------------------------------------


def test_warmup_imports_is_strict():
    sys.modules.pop("wave", None)
    wu.warmup_imports("wave")
    assert "wave" in sys.modules
    with pytest.raises(ModuleNotFoundError):
        wu.warmup_imports("this_module_does_not_exist_xyz")


def test_warmup_imports_best_effort_logs_and_records(caplog):
    report = wu.WarmupReport()
    with caplog.at_level(logging.WARNING, logger="tabarena.models.warmup"):
        imported = wu.warmup_imports_best_effort("wave", "this_module_does_not_exist_xyz", "colorsys", report=report)
    assert imported == ["wave", "colorsys"]
    assert "this_module_does_not_exist_xyz" in caplog.text
    assert report.steps == [
        "import:wave",
        "import:this_module_does_not_exist_xyz:failed:ModuleNotFoundError",
        "import:colorsys",
    ]
    assert report.failed_steps == ["import:this_module_does_not_exist_xyz:failed:ModuleNotFoundError"]


def test_report_to_dict_contains_every_field_and_dummy_fit():
    report = wu.WarmupReport(duration_s=1.5)
    report.step("x")
    report.step("y", failed=True, error=ValueError("bad"))
    out = report.to_dict()
    assert out["steps"] == ["x", "y:failed:ValueError"]
    assert out["failed_steps"] == ["y:failed:ValueError"]
    assert out["duration_s"] == 1.5
    assert out["dummy_fit"] is None and out["dummy_fits"] == []
    assert report.dummy_fit is None


# --- dispatch -----------------------------------------------------------------------------------


def test_declared_warmup_classmethod_gets_full_context():
    _DeclaredWarmupModel.calls.clear()
    report = wu.warmup_model_cls(
        _DeclaredWarmupModel, problem_type="binary", num_cpus=4, num_gpus=1, hyperparameters={"lr": 0.1}
    )
    assert _DeclaredWarmupModel.calls == [
        {"problem_type": "binary", "num_cpus": 4, "num_gpus": 1, "hyperparameters": {"lr": 0.1}}
    ]
    assert report.model_classes == ["_DeclaredWarmupModel"]
    assert report.steps == ["warmup:_DeclaredWarmupModel", "dummy_fit:_DeclaredWarmupModel:skipped"]
    assert report.dummy_fit["skipped_reason"] == "warmup_dummy_fit is False"


def test_torch_model_gets_generic_torch_warmup(no_torch_warmup):
    report = wu.warmup_model_cls(_TorchModel)
    assert no_torch_warmup == [{"cuda": None}]  # num_gpus unknown, so the torch layer auto-detects CUDA
    assert "torch:import" in report.steps


def test_num_gpus_gates_cuda_warmup(no_torch_warmup):
    wu.warmup_model_cls(_TorchModel, num_gpus=0)
    wu.warmup_model_cls(_TorchModel, num_gpus=1)
    assert no_torch_warmup == [{"cuda": False}, {"cuda": True}]


def test_classmethod_runs_before_generic_torch_layer(monkeypatch):
    _DeclaredTorchModel.order.clear()

    def fake(**kw):
        _DeclaredTorchModel.order.append("torch")
        return ["torch:import"]

    monkeypatch.setattr(wu, "warmup_torch", fake)
    report = wu.warmup_model_cls(_DeclaredTorchModel)
    assert _DeclaredTorchModel.order == ["classmethod", "torch"]
    assert report.steps[:2] == ["warmup:_DeclaredTorchModel", "torch:import"]


def test_classmethod_exception_does_not_skip_generic_layers(no_torch_warmup, caplog):
    with caplog.at_level(logging.WARNING, logger="tabarena.models.warmup"):
        report = wu.warmup_model_cls(_RaisingWarmupModel)
    assert no_torch_warmup == [{"cuda": None}]
    assert report.steps[0] == "warmup:_RaisingWarmupModel:failed:RuntimeError"
    assert report.failed_steps == ["warmup:_RaisingWarmupModel:failed:RuntimeError"]
    assert "classmethod bug" in caplog.text


def test_warmup_modules_classvar_merged_across_mro_base_first():
    assert wu.collect_warmup_modules(_ChildWithModules) == ("wave", "this_module_does_not_exist_xyz", "colorsys")
    assert wu.collect_warmup_modules(_PlainModel) == ()


def test_failing_warmup_module_is_logged_not_raised(no_torch_warmup, caplog):
    sys.modules.pop("colorsys", None)
    with caplog.at_level(logging.WARNING, logger="tabarena.models.warmup"):
        report = wu.warmup_model_cls(_ChildWithModules)
    assert "this_module_does_not_exist_xyz" in caplog.text
    assert "import:this_module_does_not_exist_xyz:failed:ModuleNotFoundError" in report.steps
    assert "import:colorsys" in report.steps and "colorsys" in sys.modules


def test_torch_entry_in_warmup_modules_routes_once(no_torch_warmup, recorded_imports):
    class _TorchBacked:
        ag_key = "MNCA_LIKE"
        warmup_dummy_fit = False
        warmup_modules = ("torch", "wave", "torch")

    wu.warmup_model_cls(_TorchBacked, num_gpus=0)
    assert no_torch_warmup == [{"cuda": False}]
    assert recorded_imports == ["wave"]


def test_ag_key_map_import_only_warmup(recorded_imports):
    report = wu.warmup_model_cls(_MappedModel)
    assert recorded_imports == ["lightgbm"]
    assert "import:lightgbm" in report.steps


def test_ag_key_map_torch_entry_routes_to_torch_warmup(no_torch_warmup, recorded_imports):
    class _NNTorchModel:
        ag_key = "NN_TORCH"
        warmup_dummy_fit = False

    wu.warmup_model_cls(_NNTorchModel, num_gpus=0)
    assert no_torch_warmup == [{"cuda": False}]
    assert recorded_imports == []


def test_ag_key_map_applies_to_torch_subclasses(no_torch_warmup, recorded_imports):
    wu.warmup_model_cls(_TabICLLike, num_gpus=0)
    assert no_torch_warmup == [{"cuda": False}]  # torch warmed once, not again for the map's "torch"
    assert recorded_imports == ["tabicl"]


def test_ag_key_extra_callable_runs(monkeypatch, recorded_imports):
    calls: list[str] = []

    def fake_native():
        calls.append("native")

    monkeypatch.setitem(wu.WARMUP_STEPS_BY_AG_KEY, "EBM", ("interpret.glassbox", fake_native))
    report = wu.warmup_model_cls(_EBMLike)
    assert calls == ["native"]
    assert recorded_imports == ["interpret.glassbox"]
    assert "extra:EBM:fake_native" in report.steps


def test_ag_key_map_covers_the_autogluon_builtins():
    expected = {
        "GBM",
        "CAT",
        "XGB",
        "EBM",
        "FASTAI",
        "NN_TORCH",
        "TABICL",
        "TABDPT",
        "TABDPT-TURBO",
        "NORI",
        "REALMLP",
        "TABPFN-3",
        "TABPFN-2.6",
        "REALTABPFN-V2",
        "REALTABPFN-V2.5",
    }
    assert expected <= set(wu.WARMUP_STEPS_BY_AG_KEY)
    from autogluon.tabular.registry import ag_model_registry

    assert expected <= set(ag_model_registry.keys)
    assert wu.warmup_ebm_native in wu.WARMUP_STEPS_BY_AG_KEY["EBM"]
    assert not hasattr(wu, "WARMUP_IMPORTS_BY_AG_KEY")


def test_unknown_model_is_a_noop(monkeypatch):
    monkeypatch.setattr(wu, "warmup_torch", lambda **kw: (_ for _ in ()).throw(AssertionError("not expected")))
    monkeypatch.setattr(
        wu, "warmup_imports_best_effort", lambda *n, **k: (_ for _ in ()).throw(AssertionError("not expected"))
    )
    report = wu.warmup_model_cls(_PlainModel)
    assert report.steps == ["dummy_fit:_PlainModel:skipped"]
    assert report.failed_steps == []


def test_warmup_model_classes_collects_into_one_report(no_torch_warmup):
    _DeclaredWarmupModel.calls.clear()
    report = wu.warmup_model_classes(
        [(_DeclaredWarmupModel, {"a": 1}), (_TorchModel, None)], problem_type="binary", num_cpus=2, num_gpus=0
    )
    assert report.model_classes == ["_DeclaredWarmupModel", "_TorchModel"]
    assert _DeclaredWarmupModel.calls[0]["hyperparameters"] == {"a": 1}
    assert no_torch_warmup == [{"cuda": False}]


# --- torch helpers -------------------------------------------------------------------------------


def test_kernel_probe_flag_reads_env(monkeypatch):
    monkeypatch.delenv(wu.KERNEL_PROBE_ENV, raising=False)
    assert wu.kernel_probe_enabled() is False
    monkeypatch.setenv(wu.KERNEL_PROBE_ENV, "1")
    assert wu.kernel_probe_enabled() is True
    monkeypatch.setenv(wu.KERNEL_PROBE_ENV, "off")
    assert wu.kernel_probe_enabled() is False


def test_warmup_torch_returns_steps_without_cuda(monkeypatch):
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    # an earlier test in the session may have created the CUDA context on a GPU host
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    assert wu.warmup_torch(cuda=None) == ["torch:import"]
    assert wu.warmup_torch(cuda=True) == ["torch:import"]  # requested but unavailable
    before = wu.snapshot_torch_globals()
    assert before is not None and set(before) == set(wu._TORCH_GLOBALS)
    assert wu.cuda_initialized() is False


def test_device_and_num_gpus_helpers(monkeypatch):
    assert wu.device_for_num_gpus(1) == "cuda"
    assert wu.device_for_num_gpus(0.5) == "cuda"
    assert wu.device_for_num_gpus(0) == "cpu"
    monkeypatch.setattr(wu, "_cuda_available", lambda: False)
    assert wu.device_for_num_gpus(None, default_num_gpus=1) == "cpu"
    monkeypatch.setattr(wu, "_cuda_available", lambda: True)
    assert wu.device_for_num_gpus(None, default_num_gpus=1) == "cuda"
    assert wu.device_for_num_gpus(None, default_num_gpus=0) == "cpu"

    assert wu.resolve_warmup_num_gpus({"ag_args_fit": {"num_gpus": 1}}, 0) == 1
    assert wu.resolve_warmup_num_gpus({"ag_args_fit": {"num_gpus": "auto"}}, 0) == 0
    assert wu.resolve_warmup_num_gpus({}, None) is None
    assert wu.strip_ag_args({"a": 1, "ag_args": {}, "ag_args_fit": {}, "ag_args_ensemble": {}}) == {"a": 1}
    assert wu.strip_ag_args(None) == {}


# --- fold fitting strategy and ray -----------------------------------------------------------------


class _SequentialDefaultModel(AbstractModel):
    ag_key = "_SEQ"
    _default_ag_args_ensemble_extra = {"fold_fitting_strategy": "sequential_local"}


class _AutoModel(AbstractModel):
    ag_key = "_AUTO"


class _NoParallelModel(AbstractModel):
    ag_key = "_NOPAR"
    _default_ag_args_ensemble_extra = {"_disable_parallel_fitting": True}


class _RayOk:
    @staticmethod
    def is_initialized():
        return False


def test_resolve_fold_fitting_strategy(monkeypatch):
    monkeypatch.setattr(ray_utils, "try_import_ray", _RayOk)
    assert wu.resolve_fold_fitting_strategy(_AutoModel, {}, problem_type="binary") == "parallel_local"
    assert wu.resolve_fold_fitting_strategy(_SequentialDefaultModel, {}, problem_type="binary") == "sequential_local"
    # The user's ag_args_ensemble wins over the class default.
    hps = {"ag_args_ensemble": {"fold_fitting_strategy": "parallel_local"}}
    assert wu.resolve_fold_fitting_strategy(_SequentialDefaultModel, hps, problem_type="binary") == "parallel_local"
    # gpu / cpu variants are selected by the bag's GPU count.
    hps = {"ag_args_ensemble": {"fold_fitting_strategy_gpu": "sequential_local", "fold_fitting_strategy_cpu": "auto"}}
    assert wu.resolve_fold_fitting_strategy(_AutoModel, hps, num_gpus=1) == "sequential_local"
    assert wu.resolve_fold_fitting_strategy(_AutoModel, hps, num_gpus=0) == "parallel_local"
    assert wu.resolve_fold_fitting_strategy(_NoParallelModel, {}) == "sequential_local"

    def no_ray():
        raise ImportError("no ray")

    monkeypatch.setattr(ray_utils, "try_import_ray", no_ray)
    assert wu.resolve_fold_fitting_strategy(_AutoModel, {}) == "sequential_local"


def test_ray_worker_warmup_modules_dedupes_and_keeps_torch_import_only():
    class _Bagged(AbstractTorchModel):
        ag_key = "TABICL"
        warmup_modules = ("tabicl", "tabicl.extra")

    modules = wu.ray_worker_warmup_modules(_Bagged)
    assert modules[:2] == wu.RAY_WORKER_WARMUP_MODULES
    assert __name__ in modules
    assert modules.count("tabicl") == 1
    assert "torch" in modules and "tabicl.extra" in modules


def test_warmup_ag_stack_records_ray_state(monkeypatch, recorded_imports):
    class _FakeRayModule:
        @staticmethod
        def is_initialized():
            return True

    monkeypatch.setitem(sys.modules, "ray", _FakeRayModule())
    report = wu.WarmupReport()
    wu.warmup_ag_stack(report=report)
    assert recorded_imports == ["autogluon.tabular"]
    assert report.steps == ["import:autogluon.tabular", "ray:already_imported", "ray:already_initialized"]
    assert report.ray["already_initialized"] is True


def test_warmup_ag_stack_without_ray(monkeypatch, recorded_imports):
    monkeypatch.delitem(sys.modules, "ray", raising=False)
    report = wu.WarmupReport()
    wu.warmup_ag_stack(report=report)
    assert report.steps == ["import:autogluon.tabular"]


# --- dummy fit ---------------------------------------------------------------------------------------


def _rng_states():
    import torch

    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _assert_rng_states_equal(before, after):
    import torch

    assert before[0] == after[0]
    assert before[1][0] == after[1][0] and np.array_equal(before[1][1], after[1][1]) and before[1][2:] == after[1][2:]
    assert torch.equal(before[2], after[2])


def test_dummy_fit_runs_with_synthetic_shapes_and_cleans_up(tmp_path):
    torch = pytest.importorskip("torch")
    _MeanModel.fits.clear()

    threads = torch.get_num_threads()
    before = _rng_states()
    report = wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=2, num_gpus=0, hyperparameters={"lr": 1})
    after = _rng_states()

    assert len(_MeanModel.fits) == 1
    fit = _MeanModel.fits[0]
    assert fit["n_rows"] == 96 and fit["n_cols"] == 6
    assert fit["dtypes"][-1] == "category"
    assert fit["num_cpus"] == 2 and fit["num_gpus"] == 0
    assert fit["time_limit"] is not None and fit["time_limit"] <= wu.DUMMY_FIT_TIME_LIMIT_S
    assert fit["params"]["lr"] == 1
    assert not __import__("os").path.exists(fit["path"])

    record = report.dummy_fit
    assert record["ran"] is True and record["error"] is None and record["skipped_reason"] is None
    assert record["n_rows"] == 96 and record["problem_type"] == "binary" and record["num_gpus"] == 0
    assert record["duration_s"] >= 0
    assert record["torch_globals_changed"] == {}
    assert "dummy_fit:_MeanModel" in report.steps and report.failed_steps == []
    _assert_rng_states_equal(before, after)
    assert torch.get_num_threads() == threads


def test_dummy_fit_is_skipped_once_the_process_is_warm():
    """The second warm-up of the same class, problem type, device kind and config skips the dummy fit."""
    _MeanModel.fits.clear()
    first = wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0, hyperparameters={"lr": 1})
    second = wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0, hyperparameters={"lr": 1})
    assert len(_MeanModel.fits) == 1
    assert first.dummy_fit["ran"] is True and first.dummy_fit["skipped_reason"] is None
    assert second.dummy_fit["ran"] is False and second.dummy_fit["skipped_reason"] == wu.ALREADY_WARM_REASON
    assert "dummy_fit:_MeanModel:skipped" in second.steps and second.failed_steps == []
    assert wu.already_warm(_MeanModel, problem_type="binary", num_gpus=0, hyperparameters={"lr": 1})


def test_warm_memo_is_keyed_on_problem_type_device_and_config():
    """A different problem type, GPU flag or model configuration warms again; AutoGluon's ag_args do not count."""
    _MeanModel.fits.clear()
    wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0, hyperparameters={"lr": 1})
    wu.warmup_model_cls(_MeanModel, problem_type="regression", num_cpus=1, num_gpus=0, hyperparameters={"lr": 1})
    wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0, hyperparameters={"lr": 2})
    assert len(_MeanModel.fits) == 3
    wu.warmup_model_cls(
        _MeanModel,
        problem_type="binary",
        num_cpus=1,
        num_gpus=0,
        hyperparameters={"lr": 1, "ag_args": {"name_suffix": "_x"}},
    )
    assert len(_MeanModel.fits) == 3
    assert wu.warm_key(_MeanModel, problem_type="binary", num_gpus=0, hyperparameters={"lr": 1}) != wu.warm_key(
        _MeanModel, problem_type="binary", num_gpus=1, hyperparameters={"lr": 1}
    )


def test_failed_dummy_fit_is_not_remembered_as_warm(caplog):
    with caplog.at_level(logging.WARNING):
        wu.warmup_model_cls(_BoomModel, problem_type="binary", num_cpus=1, num_gpus=0)
        second = wu.warmup_model_cls(_BoomModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert second.dummy_fit["skipped_reason"] is None and second.dummy_fit["error"] is not None
    assert not wu.already_warm(_BoomModel, problem_type="binary", num_gpus=0, hyperparameters=None)


def test_warmup_always_env_disables_the_skip(monkeypatch):
    _MeanModel.fits.clear()
    monkeypatch.setenv(wu.WARMUP_ALWAYS_ENV, "1")
    wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert len(_MeanModel.fits) == 2


def test_cpu_dummy_fit_never_forks_cuda_generators_on_a_cuda_host(monkeypatch):
    """``num_gpus=0`` forks only the CPU generator even when CUDA is available, so no context is created."""
    torch = pytest.importorskip("torch")

    calls: list[list[int]] = []
    original_fork_rng = torch.random.fork_rng

    def recording_fork_rng(*args, **kwargs):
        calls.append(list(kwargs.get("devices") or []))
        return original_fork_rng(*args, **kwargs)

    monkeypatch.setattr(torch.random, "fork_rng", recording_fork_rng)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: False)
    _MeanModel.fits.clear()
    report = wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert report.dummy_fit["ran"] is True and report.dummy_fit["num_gpus"] == 0
    assert calls == [[]]


def test_dummy_fit_records_torch_globals_and_restores_threads():
    """A plain ``AbstractModel`` has no torch layer, so the dummy fit itself must record the torch globals."""
    torch = pytest.importorskip("torch")

    class _ThreadHogModel(_MeanModel):
        ag_key = "_WARMUP_MEAN_THREADS"

        def _fit(self, X, y, **kwargs):
            torch.set_num_threads(max(1, torch.get_num_threads() - 1) if torch.get_num_threads() > 1 else 2)
            super()._fit(X, y, **kwargs)

    threads = torch.get_num_threads()
    report = wu.warmup_model_cls(_ThreadHogModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert report.dummy_fit["ran"] is True
    assert report.torch_globals_before is not None
    assert report.torch_globals_before["num_threads"] == threads
    changed = report.dummy_fit["torch_globals_changed"]
    assert list(changed) == ["num_threads"] and changed["num_threads"][0] == threads
    assert changed["num_threads"][1] != threads
    assert torch.get_num_threads() == threads


@pytest.mark.parametrize("problem_type", ["multiclass", "regression"])
def test_dummy_fit_predicts_per_problem_type(problem_type):
    _MeanModel.fits.clear()
    report = wu.warmup_model_cls(_MeanModel, problem_type=problem_type, num_cpus=1, num_gpus=0)
    assert report.dummy_fit["ran"] is True, report.dummy_fit
    assert len(_MeanModel.fits) == 1


def test_dummy_fit_applies_class_overrides():
    _CheapMeanModel.fits.clear()
    report = wu.warmup_model_cls(
        _CheapMeanModel, problem_type="binary", num_cpus=1, num_gpus=0, hyperparameters={"lr": 2}
    )
    fit = _CheapMeanModel.fits[0]
    assert fit["n_rows"] == 40 and fit["n_cols"] == 4
    assert fit["dtypes"].count("category") == 2
    assert fit["time_limit"] <= 7
    assert fit["params"]["n_estimators"] == 1 and fit["params"]["lr"] == 2
    assert report.dummy_fit["n_rows"] == 40


def test_dummy_fit_failure_is_recorded_not_raised(caplog):
    with caplog.at_level(logging.WARNING, logger="tabarena.models.warmup"):
        report = wu.warmup_model_cls(_BoomModel, problem_type="regression", num_cpus=1, num_gpus=0)
    record = report.dummy_fit
    assert record["ran"] is False
    assert "fit exploded" in record["error"] and "RuntimeError" in record["error"]
    assert report.failed_steps == ["dummy_fit:_BoomModel:failed:RuntimeError"]
    assert "fit exploded" in caplog.text


def test_dummy_fit_gates(monkeypatch):
    _GpuOnlyMeanModel.fits.clear()
    report = wu.warmup_model_cls(_GpuOnlyMeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert _GpuOnlyMeanModel.fits == []
    assert report.dummy_fit["skipped_reason"] == "model needs a GPU and none is allocated"

    # A model that only prefers a GPU (LightGBM-style minimum_num_gpus=0.5) fits on the CPU when no
    # CUDA device exists, and is skipped only when a GPU exists but none was allocated.
    _GpuPreferringMeanModel.fits.clear()
    monkeypatch.setattr(wu, "_cuda_available", lambda: False)
    report = wu.warmup_model_cls(_GpuPreferringMeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert len(_GpuPreferringMeanModel.fits) == 1 and report.dummy_fit["ran"] is True
    monkeypatch.setattr(wu, "_cuda_available", lambda: True)
    report = wu.warmup_model_cls(_GpuPreferringMeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert len(_GpuPreferringMeanModel.fits) == 1
    assert report.dummy_fit["skipped_reason"] == "model needs a GPU and none is allocated"
    monkeypatch.undo()

    _RegressionOnlyMeanModel.fits.clear()
    report = wu.warmup_model_cls(_RegressionOnlyMeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert _RegressionOnlyMeanModel.fits == []
    assert "not supported" in report.dummy_fit["skipped_reason"]

    _OptedOutMeanModel.fits.clear()
    report = wu.warmup_model_cls(_OptedOutMeanModel, problem_type="binary", num_cpus=1, num_gpus=0)
    assert _OptedOutMeanModel.fits == []
    assert report.dummy_fit["skipped_reason"] == "warmup_dummy_fit is False"

    _MeanModel.fits.clear()
    report = wu.warmup_model_cls(_MeanModel, problem_type=None, num_cpus=1, num_gpus=0)
    assert _MeanModel.fits == [] and report.dummy_fit["skipped_reason"] == "problem_type unknown"

    report = wu.warmup_model_cls(_MeanModel, problem_type="binary", num_cpus=1, num_gpus=0, dummy_fit=False)
    assert _MeanModel.fits == [] and report.dummy_fits == []
    assert "dummy_fit:_MeanModel:skipped" in report.steps


# --- feature generators and run_warmup_fn ----------------------------------------------------------------


def test_warmup_feature_generator_cls_dispatches_classvar_and_classmethod(recorded_imports):
    calls: list[dict] = []

    class _Gen:
        warmup_modules = ("skrub_like",)

        @classmethod
        def warmup(cls, *, feature_generator_kwargs=None, **kwargs):
            calls.append({"feature_generator_kwargs": feature_generator_kwargs})

    report = wu.warmup_feature_generator_cls(_Gen, {"enable_x": False})
    assert recorded_imports == ["skrub_like"]
    assert calls == [{"feature_generator_kwargs": {"enable_x": False}}]
    assert report.steps == ["import:skrub_like", "warmup:_Gen"]
    assert wu.warmup_feature_generator_cls(None).steps == []


def test_run_warmup_fn_records_imports_and_status(throwaway_package, monkeypatch):
    # an earlier test in the session may have created the CUDA context on a GPU host
    monkeypatch.setattr(wu, "cuda_initialized", lambda: False)

    def fn():
        __import__(throwaway_package)

    report = wu.run_warmup_fn(fn, label="M")
    assert report.status == "ok" and report.label == "M"
    assert report.duration_s is not None and report.duration_s >= 0
    assert throwaway_package in report.imported_modules
    assert report.cuda_initialized in (None, False)

    def partial():
        rep = wu.WarmupReport()
        rep.step("import:x", failed=True, error=ImportError("x"))
        return rep

    assert wu.run_warmup_fn(partial, label="M").status == "partial"


def test_run_warmup_fn_failure_prints_header_and_traceback(capsys):
    def boom():
        raise RuntimeError("warm-up bug")

    report = wu.run_warmup_fn(boom, label="MyMethod")
    assert report.status == "failed" and report.duration_s is None
    assert report.error == "RuntimeError: warm-up bug"
    out = capsys.readouterr()
    header = "Warm-up of method 'MyMethod' failed (fitting cold instead):"
    assert header in out.out
    assert "RuntimeError: warm-up bug" in out.err
