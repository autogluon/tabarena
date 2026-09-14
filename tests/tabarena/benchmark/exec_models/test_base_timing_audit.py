"""Tests for ``AbstractExecModel.fit_custom``: frame ownership, timer bracketing and the timing audit.

A recording exec model with trivial ``_fit`` / ``_predict`` stands in for a real method; no
AutoGluon model is fit. Everything runs on the CPU, offline.
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd
import pytest
from autogluon.core.metrics import get_metric

from tabarena.benchmark.exec_models.base import TIMING_AUDIT_SCOPE, AbstractExecModel

_DIFF_KEYS = {
    "new_modules",
    "new_packages",
    "new_submodule_packages",
    "cuda_initialized_before",
    "cuda_initialized_after",
    "ray_initialized_before",
    "ray_initialized_after",
}
_MEMORY_KEYS = {
    "peak_mem_cpu",
    "min_mem_cpu",
    "peak_mem_gpu",
    "peak_mem_gpu_reserved",
    "min_mem_gpu",
    "min_mem_gpu_reserved",
    "gpu_tracking_enabled",
    "baseline_mem_cpu",
    "baseline_mem_cpu_self",
    "cpu_tracking_backend",
}


class _RecordingExecModel(AbstractExecModel):
    """Records what ``_fit`` received and mutates the frame it was given."""

    preprocess_data = False
    preprocess_label = False

    def __init__(self, *args, fit_hook=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.fit_X = None
        self.fit_y = None
        self.in_place_during_fit = None
        self._fit_hook = fit_hook

    def _fit(self, X, y, **kwargs):
        self.fit_X = X
        self.fit_y = y
        self.in_place_during_fit = self._can_use_data_in_place
        X["appended_label"] = y  # a wrapper attaching the label in place, like AGWrapper does
        if self._fit_hook is not None:
            self._fit_hook()
        return self

    def _predict(self, X):
        return pd.Series(np.zeros(len(X)), index=X.index)

    def _predict_proba(self, X):
        return pd.DataFrame({0: np.full(len(X), 0.25), 1: np.full(len(X), 0.75)}, index=X.index)


def _frames(n: int = 12):
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(n), "b": rng.standard_normal(n)})
    y = pd.Series(rng.standard_normal(n), name="target")
    X_test = pd.DataFrame({"a": rng.standard_normal(5), "b": rng.standard_normal(5)})
    return X, y, X_test


def _model(problem_type: str = "regression", **kwargs) -> _RecordingExecModel:
    metric = "rmse" if problem_type == "regression" else "log_loss"
    return _RecordingExecModel(
        problem_type=problem_type, eval_metric=get_metric(metric, problem_type=problem_type), **kwargs
    )


def test_fit_custom_returns_timing_audit_and_memory_keys():
    X, y, X_test = _frames()
    out = _model().fit_custom(X, y, X_test)

    assert set(out) == {"predictions", "probabilities", "time_train_s", "time_infer_s", "memory_usage", "timing_audit"}
    audit = out["timing_audit"]
    assert set(audit) == {"fit", "predict", "scope"}
    assert audit["scope"] == TIMING_AUDIT_SCOPE
    assert set(audit["fit"]) == _DIFF_KEYS
    assert set(audit["predict"]) == _DIFF_KEYS
    memory = out["memory_usage"]
    assert set(memory) == _MEMORY_KEYS
    assert memory["baseline_mem_cpu"] >= memory["baseline_mem_cpu_self"] > 0
    assert memory["cpu_tracking_backend"] in {"procfs", "psutil"}
    assert out["time_train_s"] >= 0 and out["time_infer_s"] >= 0


def test_fit_custom_classification_audits_the_predict_proba_timer():
    X, y, X_test = _frames()
    y = pd.Series(np.arange(len(X)) % 2)
    out = _model("binary").fit_custom(X, y, X_test)
    assert out["probabilities"].shape == (len(X_test), 2)
    assert set(out["timing_audit"]["predict"]) == _DIFF_KEYS


def test_fit_custom_copies_the_caller_frame_before_the_fit():
    X, y, X_test = _frames()
    columns_before = list(X.columns)
    model = _model()
    assert model._can_use_data_in_place is False

    model.fit_custom(X, y, X_test)

    assert model.in_place_during_fit is True  # the fit worked on an owned frame ...
    assert model.fit_X is not X  # ... which was a copy taken by fit_custom ...
    assert list(X.columns) == columns_before  # ... so the caller's frame is untouched
    assert model._can_use_data_in_place is False  # the flag is restored afterwards


def test_fit_custom_restores_the_in_place_flag_when_the_fit_raises():
    X, y, X_test = _frames()

    def boom():
        raise RuntimeError("fit failed")

    model = _model(fit_hook=boom)
    with pytest.raises(RuntimeError, match="fit failed"):
        model.fit_custom(X, y, X_test)
    assert model._can_use_data_in_place is False


def test_fit_custom_shuffle_features_makes_no_second_copy(monkeypatch):
    X, y, X_test = _frames()
    columns = list(X.columns)
    shuffled = X[list(reversed(columns))].copy()  # standalone, like a frame the shuffle owns
    model = _model()
    model.shuffle_features = True
    monkeypatch.setattr(model, "_shuffle_features", lambda X, *, split_seed: (shuffled, list(reversed(columns))))

    model.fit_custom(X, y, X_test, split_seed=3)

    assert model.fit_X is shuffled  # the shuffle's new frame is used as is
    assert list(X.columns) == columns


def test_fit_custom_lazy_loaded_frames_are_owned_and_not_copied():
    X, y, X_test = _frames()
    loads: list[tuple] = []

    def lazy_load():
        frames = (X.copy(), y.copy(), X_test.copy())
        loads.append(frames)
        return frames

    model = _model()
    model.fit_custom(None, None, None, lazy_load_function=lazy_load)

    assert len(loads) == 2  # once for the fit, once for inference
    assert model.fit_X is loads[0][0]  # the loaded frame itself, no copy
    assert model.in_place_during_fit is True
    assert model._can_use_data_in_place is False


def test_timing_audit_attributes_a_cold_import_to_the_fit(tmp_path, monkeypatch):
    package = tmp_path / "fakepkg_fit_probe"
    package.mkdir()
    (package / "__init__.py").write_text("VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("fakepkg_fit_probe", None)

    def cold_import():
        import fakepkg_fit_probe  # noqa: F401

    X, y, X_test = _frames()
    try:
        out = _model(fit_hook=cold_import).fit_custom(X, y, X_test)
    finally:
        sys.modules.pop("fakepkg_fit_probe", None)

    assert "fakepkg_fit_probe" in out["timing_audit"]["fit"]["new_packages"]
    assert "fakepkg_fit_probe" not in out["timing_audit"]["predict"]["new_packages"]


def test_bag_artifact_signature_accepts_timed_outputs():
    model = _model()
    with pytest.raises(NotImplementedError):
        model.bag_artifact(pd.DataFrame(), y_pred=None, y_pred_proba=None)


def test_uses_ray_defaults_to_true():
    assert AbstractExecModel.uses_ray({}) is True
    assert _RecordingExecModel.uses_ray({"fit_kwargs": {}}, problem_type="binary") is True
