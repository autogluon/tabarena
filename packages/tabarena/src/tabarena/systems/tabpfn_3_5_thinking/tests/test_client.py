"""The TabPFN-3.5-Thinking wrapper against a fake client: arguments, retries and the recorded metadata (nothing is sent)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

tabpfn_client = pytest.importorskip("tabpfn_client")

from autogluon.core.metrics import get_metric
from tabpfn_client.errors import RetryableServerError

from tabarena.systems.tabpfn_3_5_thinking import system as thinking
from tabarena.systems.tabpfn_3_5_thinking.system import TabPFN35ThinkingSystemModel


class _FakeEstimator:
    """Records what the wrapper passes; ``failures`` holds the exceptions the next fits raise, in order."""

    created: list[_FakeEstimator] = []
    failures: list[BaseException] = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.fit_calls: list[tuple[pd.DataFrame, pd.Series]] = []
        self.predict_calls: list[pd.DataFrame] = []
        self._last_meta = {}
        _FakeEstimator.created.append(self)

    def fit(self, X, y):
        self.fit_calls.append((X.copy(), y.copy()))
        if _FakeEstimator.failures:
            raise _FakeEstimator.failures.pop(0)
        self.classes_ = np.array(sorted(pd.unique(y)))
        self.model_id_ = "fit-1"
        self.fit_timings_ = {"elapsed_s": 5.0, "queue_wait_s": 0.5, "train_set_transform_s": 0.1, "fit_s": 4.2}
        return self

    def _predicted(self, X):
        self.predict_calls.append(X.copy())
        self._last_meta = {"package_version": "0.0.0"}

    def predict_proba(self, X):
        self._predicted(X)
        return np.full((len(X), len(self.classes_)), 1 / len(self.classes_))

    def predict(self, X):
        self._predicted(X)
        return np.zeros(len(X))

    def get_timings(self):
        return {
            "fit": getattr(self, "fit_timings_", None),
            "predict": {"predict_s": 0.3} if self.predict_calls else None,
        }


@pytest.fixture
def fake_client(monkeypatch):
    _FakeEstimator.created, _FakeEstimator.failures = [], []
    tokens = []
    monkeypatch.setattr(tabpfn_client, "TabPFNClassifier", _FakeEstimator)
    monkeypatch.setattr(tabpfn_client, "TabPFNRegressor", _FakeEstimator)
    monkeypatch.setattr(tabpfn_client, "set_access_token", tokens.append)
    monkeypatch.setattr(thinking.time, "sleep", lambda _s: None)
    monkeypatch.setattr(thinking, "_model_limit", _limit)
    monkeypatch.setattr(thinking, "_timeout_cap", lambda _effort: 4 * 3600.0)
    monkeypatch.setenv("TABPFN_TOKEN", "test-key")
    return tokens


def _limit(test_rows: int = 10**6) -> SimpleNamespace:
    return SimpleNamespace(test_set_max_rows=test_rows, predict_row_pairs_budget=10**12, test_set_max_cells=10**9)


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": pd.Categorical(rng.choice(["x", "y"], size=40))})
    y = pd.Series(np.where(X["a"] > 0, "pos", "neg"), name="label")
    return X.iloc[:30], y.iloc[:30], X.iloc[30:]


def _model(problem_type: str = "binary", metric: str = "roc_auc", time_limit: float = 60, **kwargs):
    kwargs.setdefault("validation_metadata", {"target_name": "label"})
    return TabPFN35ThinkingSystemModel(
        problem_type=problem_type,
        eval_metric=get_metric(metric, problem_type=problem_type),
        fit_kwargs={"time_limit": time_limit},
        **kwargs,
    )


def test_fit_forwards_effort_metric_seed_and_time_limit(fake_client, frames):
    X, y, X_test = frames
    model = _model(effort="high")
    out = model.fit_custom(X, y, X_test, split_seed=3)

    assert out["probabilities"].shape == (len(X_test), 2)
    assert fake_client == ["test-key"]  # the key from the environment, never a cached token
    (est,) = _FakeEstimator.created
    assert est.kwargs == {
        "thinking_effort": "high",
        "thinking_timeout_s": 60.0,  # the task's time limit
        "thinking_metric": "roc_auc",
        "random_state": 3,  # the split's random_state
    }
    ((fit_X, fit_y),), (pred_X,) = est.fit_calls, est.predict_calls
    assert "label" not in fit_X.columns
    assert "label" not in pred_X.columns
    assert len(pred_X) == len(X_test)
    assert list(fit_y) == list(y)


def test_regression_uses_the_regressor_and_its_metric(fake_client, monkeypatch, frames):
    X, _, X_test = frames
    regressors = []

    class _Regressor(_FakeEstimator):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            regressors.append(self)

    monkeypatch.setattr(tabpfn_client, "TabPFNRegressor", _Regressor)
    y = pd.Series(np.arange(len(X), dtype=float), name="label")
    out = _model("regression", "rmse").fit_custom(X, y, X_test, split_seed=0)

    assert len(regressors) == 1
    assert regressors[0].kwargs["thinking_metric"] == "root_mean_squared_error"
    assert out["probabilities"] is None
    assert len(out["predictions"]) == len(X_test)


@pytest.mark.parametrize(
    ("time_limit", "cap", "sent"),
    [(4 * 3600, 3600.0, 3600.0), (60, None, 60.0), (None, 3600.0, None)],
)
def test_time_limit_is_capped_at_the_effort_maximum(fake_client, monkeypatch, frames, time_limit, cap, sent):
    X, y, X_test = frames
    monkeypatch.setattr(thinking, "_timeout_cap", lambda _effort: cap)
    model = _model(time_limit=time_limit)
    model.fit_custom(X, y, X_test, split_seed=0)

    assert _FakeEstimator.created[0].kwargs["thinking_timeout_s"] == sent
    assert model.get_metadata()["thinking_timeout_s"] == sent


def test_timeout_cap_reads_the_effort_maximum_from_the_settings(monkeypatch):
    from tabpfn_client.client import ServiceClient

    settings = SimpleNamespace(
        thinking_default_model_version="v3.5",
        thinking_model_limits={"v3.5": SimpleNamespace(max_timeout_secs={"xhigh": 14400.0})},
        thinking_max_timeout_secs=57600.0,
    )
    monkeypatch.setattr(ServiceClient, "get_settings", staticmethod(lambda: settings))
    assert (thinking._timeout_cap("xhigh"), thinking._timeout_cap("other")) == (14400.0, 57600.0)
    monkeypatch.setattr(ServiceClient, "get_settings", staticmethod(lambda: None))
    assert thinking._timeout_cap("xhigh") is None


def test_group_and_time_columns_are_named_to_the_client(fake_client, frames):
    X, y, X_test = frames
    X, X_test = X.assign(g=np.arange(len(X)) // 3, t=np.arange(len(X))), X_test.assign(g=0, t=0)
    model = _model(validation_metadata={"target_name": "label", "group_on": "g", "group_time_on": "t"})
    model.fit_custom(X, y, X_test, split_seed=0)

    (est,) = _FakeEstimator.created
    assert (est.kwargs["group_col"], est.kwargs["group_time_col"]) == ("g", "t")
    assert "time_col" not in est.kwargs
    assert {"g", "t"} <= set(est.fit_calls[0][0].columns)  # the columns themselves stay in X
    assert model.get_metadata()["api_split_columns"] == {"group_col": "g", "group_time_col": "t"}


def test_object_integer_labels_are_converted(fake_client, frames):
    X, _, X_test = frames
    y = pd.Series([int(v) for v in np.arange(len(X)) % 2], dtype=object, name="label")
    out = _model().fit_custom(X, y, X_test, split_seed=0)

    ((_, fit_y),) = _FakeEstimator.created[0].fit_calls
    assert fit_y.dtype == np.int64
    assert out["probabilities"].shape == (len(X_test), 2)


def test_large_predictions_are_sent_in_row_blocks(fake_client, monkeypatch, frames):
    X, y, X_test = frames
    monkeypatch.setattr(thinking, "_model_limit", lambda: _limit(test_rows=4))
    model = _model()
    out = model.fit_custom(X, y, X_test, split_seed=0)

    assert [len(block) for block in _FakeEstimator.created[0].predict_calls] == [4, 4, 2]
    assert out["probabilities"].shape == (len(X_test), 2)
    assert [c["rows"] for c in model.get_metadata()["api_calls"]] == [len(X), 4, 4, 2]


def test_transient_errors_are_retried_and_recorded(fake_client, frames):
    X, y, X_test = frames
    _FakeEstimator.failures = [RetryableServerError("503"), RetryableServerError("503")]
    model = _model()
    model.fit_custom(X, y, X_test, split_seed=0)

    assert len(_FakeEstimator.created[0].fit_calls) == 3
    meta = model.get_metadata()
    fit_call, predict_call = meta["api_calls"]
    assert (fit_call["call"], fit_call["attempts"], fit_call["failure"]) == ("fit", 3, "RetryableServerError: 503")
    assert (predict_call["call"], predict_call["attempts"]) == ("predict_proba", 1)
    assert (meta["api_fit_s"], meta["api_predict_s"]) == (4.2, 0.3)
    assert meta["api_server_info"] == {"package_version": "0.0.0"}
    assert (meta["effort"], meta["thinking_metric"], meta["api_model_id"]) == ("xhigh", "roc_auc", "fit-1")
    assert meta["client"].startswith("tabpfn-client ")


def test_retries_are_bounded(fake_client, frames):
    X, y, X_test = frames
    _FakeEstimator.failures = [RetryableServerError("503")] * (thinking._MAX_ATTEMPTS + 1)
    with pytest.raises(RuntimeError, match=f"fit failed after {thinking._MAX_ATTEMPTS} attempts"):
        _model().fit_custom(X, y, X_test, split_seed=0)
    assert len(_FakeEstimator.created[0].fit_calls) == thinking._MAX_ATTEMPTS


def test_other_errors_fail_at_once(fake_client, frames):
    X, y, X_test = frames
    _FakeEstimator.failures = [RuntimeError("HTTP 403: forbidden")]
    with pytest.raises(RuntimeError, match="HTTP 403"):
        _model().fit_custom(X, y, X_test, split_seed=0)
    assert len(_FakeEstimator.created[0].fit_calls) == 1


def test_missing_key_fails_before_any_request(fake_client, monkeypatch, frames):
    X, y, X_test = frames
    monkeypatch.delenv("TABPFN_TOKEN")
    with pytest.raises(RuntimeError, match="TABPFN_TOKEN"):
        _model().fit_custom(X, y, X_test, split_seed=0)
    assert _FakeEstimator.created == []
    assert fake_client == []
