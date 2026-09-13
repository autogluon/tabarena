from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd
import pytest
from autogluon.core.utils.exceptions import TimeLimitExceeded

from tabarena.models.aplr.model import APLRModel


class _RecordingAPLR:
    """Stand-in for aplr's estimators that records the constructor arguments and skips the fit."""

    instances: list[_RecordingAPLR] = []

    def __init__(self, *, n_jobs: int = 0, time_limit: float = float("nan"), **params):
        self.n_jobs = n_jobs
        self.time_limit = time_limit
        self.params = params
        self.fit_kwargs: dict | None = None
        _RecordingAPLR.instances.append(self)

    def fit(self, X, y, **kwargs):
        self.fit_kwargs = kwargs
        return self

    def predict(self, X):
        return np.zeros(len(X))

    def predict_proba(self, X):
        return np.full((len(X), 2), 0.5)


class _LegacyAPLR(_RecordingAPLR):
    """An aplr release without the ``time_limit`` constructor parameter."""

    def __init__(self, *, n_jobs: int = 0, **params):
        super().__init__(n_jobs=n_jobs, **params)


def _fit(monkeypatch, tmp_path, *, stub_cls=_RecordingAPLR, problem_type="regression", with_val=False, **fit_kwargs):
    import aplr

    monkeypatch.setattr(aplr, "APLRRegressor", stub_cls)
    monkeypatch.setattr(aplr, "APLRClassifier", stub_cls)
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.standard_normal(40), "b": rng.integers(0, 3, 40).astype(float)})
    y = pd.Series(rng.standard_normal(40)) if problem_type == "regression" else pd.Series(rng.integers(0, 2, 40))
    if with_val:
        fit_kwargs["X_val"], fit_kwargs["y_val"] = X.iloc[30:], y.iloc[30:]
        X, y = X.iloc[:30], y.iloc[:30]
    model = APLRModel(path=str(tmp_path), name="aplr", problem_type=problem_type, hyperparameters={})
    model.fit(X=X, y=y, num_cpus=2, **fit_kwargs)
    return _RecordingAPLR.instances[-1]


@pytest.mark.parametrize("problem_type", ["regression", "binary"])
def test_fit_budget_is_forwarded_with_headroom(monkeypatch, tmp_path, problem_type):
    """APLR receives the remaining budget minus the wrapper's headroom, not AutoGluon's raw limit."""
    stub = _fit(monkeypatch, tmp_path, problem_type=problem_type, time_limit=100.0)
    # AutoGluon's pre-fit work is charged against the budget and can take seconds on a busy machine,
    # so only bound the deduction loosely.
    assert 100.0 * APLRModel._TIME_LIMIT_FRACTION - 30.0 < stub.time_limit <= 100.0 * APLRModel._TIME_LIMIT_FRACTION


def test_without_budget_aplr_keeps_its_default(monkeypatch, tmp_path):
    stub = _fit(monkeypatch, tmp_path)
    assert np.isnan(stub.time_limit)


def test_legacy_aplr_without_time_limit_support_warns(monkeypatch, tmp_path, caplog):
    """An aplr release that cannot take a budget must not break the fit, but the gap is logged."""
    with caplog.at_level(logging.WARNING, logger="tabarena.models.aplr.model"):
        stub = _fit(monkeypatch, tmp_path, stub_cls=_LegacyAPLR, time_limit=100.0)
    assert np.isnan(stub.time_limit)
    assert any("time_limit" in record.getMessage() for record in caplog.records)


def test_validation_split_is_passed_as_cv_observations(monkeypatch, tmp_path):
    stub = _fit(monkeypatch, tmp_path, with_val=True, time_limit=100.0)
    cv_observations = stub.fit_kwargs["cv_observations"]
    assert cv_observations.shape == (40, 1)
    assert (cv_observations[:30] == 1).all() and (cv_observations[30:] == -1).all()


def test_exhausted_budget_raises_instead_of_fitting():
    with pytest.raises(TimeLimitExceeded):
        APLRModel._aplr_time_limit(_RecordingAPLR, time_limit=1.0, time_start=time.time() - 5.0)


@pytest.mark.models
def test_real_fit_ends_within_the_budget(tmp_path):
    """With the patched aplr installed, a bagged-child fit through the wrapper honors the budget."""
    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.standard_normal((20_000, 20)), columns=[f"x{i}" for i in range(20)])
    y = pd.Series(X["x0"] * X["x1"] + np.sin(X["x2"]) + 0.1 * rng.standard_normal(len(X)))
    model = APLRModel(path=str(tmp_path), name="aplr", problem_type="regression", hyperparameters={})
    # AutoGluon's own pre-fit work (memory estimate, metadata) takes a few seconds on this frame
    # and is charged against the budget, so keep the budget well above it.
    budget = 12.0
    start = time.time()
    model.fit(X=X, y=y, time_limit=budget, num_cpus=4)
    elapsed = time.time() - start
    assert elapsed < 1.5 * budget
    assert model.model.time_limit <= budget * APLRModel._TIME_LIMIT_FRACTION
    assert model.model.get_optimal_m() < model.model.m
    assert np.isfinite(model.predict(X.head(10))).all()
