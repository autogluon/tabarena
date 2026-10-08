"""The Chakra-Tab client against a fake endpoint: payload, retries and the recorded metadata (nothing is sent)."""

from __future__ import annotations

import base64
import io

import numpy as np
import pandas as pd
import pytest

requests = pytest.importorskip("requests")
pytest.importorskip("pyarrow")

from autogluon.core.metrics import get_metric

from tabarena.systems.chakra_tab import system as chakra
from tabarena.systems.chakra_tab.system import ChakraTabSystemModel


class _Response:
    def __init__(self, status: int, payload: dict | None):
        self.status_code = status
        self._payload = payload
        self.text = "fake"

    def json(self):
        if self._payload is None:
            raise ValueError("not JSON")
        return self._payload


@pytest.fixture
def frames():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"a": rng.normal(size=40), "b": rng.choice(["x", "y"], size=40)})
    y = pd.Series(np.where(X["a"] > 0, "pos", "neg"), name="label")
    return X.iloc[:30], y.iloc[:30], X.iloc[30:]


def _model() -> ChakraTabSystemModel:
    return ChakraTabSystemModel(
        problem_type="binary",
        eval_metric=get_metric("roc_auc", problem_type="binary"),
        validation_metadata={"target_name": "label"},
        fit_kwargs={"time_limit": 60},
        preset="medium",
    )


def _ok(n: int) -> _Response:
    proba = np.full((n, 2), 0.5).tolist()
    return _Response(200, {"classes": ["neg", "pos"], "probabilities": proba, "fit": {"model": "chakra-test"}})


def test_payload_retries_and_metadata(frames, monkeypatch):
    X, y, X_test = frames
    monkeypatch.setenv("CHAKRA_TAB_KEY", "test-key")
    monkeypatch.setattr(chakra.time, "sleep", lambda _s: None)
    sent = []
    replies = iter([_Response(503, {"error": "busy"}), _Response(200, None), _ok(len(X_test))])

    def post(url, json, headers, timeout):
        sent.append(json)
        return next(replies)

    monkeypatch.setattr(requests, "post", post)
    model = _model()
    out = model.fit_custom(X, y, X_test, split_seed=0)

    assert out["probabilities"].shape == (len(X_test), 2)
    assert len(sent) == 3
    body = sent[0]
    assert set(body) == {"op", "train", "target", "test", "preset", "problem_type", "eval_metric", "time_limit"}
    test_table = pd.read_parquet(io.BytesIO(base64.b64decode(body["test"]["bytes"])))
    train_table = pd.read_parquet(io.BytesIO(base64.b64decode(body["train"]["bytes"])))
    assert "label" not in test_table.columns
    assert "label" in train_table.columns
    meta = model.get_metadata()
    assert meta["api_fit_info"] == {"model": "chakra-test"}
    assert meta["api_calls"][0]["attempts"] == 3


def test_client_errors_are_not_retried(frames, monkeypatch):
    X, y, X_test = frames
    monkeypatch.setenv("CHAKRA_TAB_KEY", "test-key")
    calls = []
    monkeypatch.setattr(requests, "post", lambda *a, **k: calls.append(1) or _Response(401, {"error": "auth"}))
    with pytest.raises(RuntimeError, match="HTTP 401"):
        _model().fit_custom(X, y, X_test, split_seed=0)
    assert len(calls) == 1


def test_missing_key_fails_before_any_request(frames, monkeypatch):
    X, y, X_test = frames
    monkeypatch.delenv("CHAKRA_TAB_KEY", raising=False)
    monkeypatch.setattr(requests, "post", lambda *a, **k: pytest.fail("no request without a key"))
    with pytest.raises(RuntimeError, match="CHAKRA_TAB_KEY"):
        _model().fit_custom(X, y, X_test, split_seed=0)
