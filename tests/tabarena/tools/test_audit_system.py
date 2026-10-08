"""The system audit must pass an honest learner and catch blatant transduction, label lookup and leaks."""

from __future__ import annotations

import base64
import importlib
import sys
import textwrap

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.exec_models import ExternalSystemModel
from tabarena.tools.audit_system import (
    FAIL,
    PASS,
    WARN,
    EgressRecorder,
    audit_system,
    compare_submitted_results,
    make_synthetic_split,
    scan_source,
)

pytest.importorskip("sklearn")


@pytest.fixture(autouse=True)
def _few_threads():
    """The fake systems fit sklearn models; OpenMP over every core of a large machine crawls."""
    from threadpoolctl import threadpool_limits

    with threadpool_limits(4):
        yield


def _encode(X: pd.DataFrame) -> np.ndarray:
    out = X.copy()
    for col in out.columns:
        if not pd.api.types.is_numeric_dtype(out[col]):
            out[col] = out[col].astype(str).map(lambda v: sum(map(ord, v)) % 7)
    return out.to_numpy(dtype=float)


class _HonestSystem(ExternalSystemModel):
    """Gradient boosting fit on the training rows only; each test row predicted on its own."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._model = None

    def _fit_system(self, X, y, *, problem_type, **kwargs):
        from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

        cls = HistGradientBoostingRegressor if problem_type == "regression" else HistGradientBoostingClassifier
        self._model = cls(random_state=0, max_iter=50).fit(self._features(X, fit=True), y)
        return self

    def _features(self, X, *, fit: bool = False) -> np.ndarray:
        return _encode(X)

    def _predict(self, X):
        return pd.Series(self._model.predict(self._features(X)), index=X.index)

    def _predict_proba(self, X):
        proba = self._model.predict_proba(self._features(X))
        return pd.DataFrame(proba, index=X.index, columns=self._model.classes_)


class _TransductiveSystem(_HonestSystem):
    """Standardizes the test batch with its own statistics: predictions depend on the other rows."""

    def _features(self, X, *, fit: bool = False) -> np.ndarray:
        values = _encode(X)
        return (values - values.mean(axis=0)) / (values.std(axis=0) + 1e-9)


class _LookupSystem(ExternalSystemModel):
    """Ignores the training labels and returns the true label of every test row it recognizes."""

    ORACLE: dict = {}

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._classes = None

    @staticmethod
    def key(row) -> tuple:
        return tuple(np.round(np.asarray(row, dtype=float), 6))

    def _fit_system(self, X, y, **kwargs):
        self._classes = sorted({str(v) for v in y})
        return self

    def _predict_proba(self, X):
        numeric = X.select_dtypes("number")
        rows = []
        for row in numeric.to_numpy():
            label = self.ORACLE.get(self.key(row))
            rows.append(
                [1.0 if c == label else 0.0 for c in self._classes]
                if label is not None
                else [1 / len(self._classes)] * len(self._classes)
            )
        return pd.DataFrame(rows, index=X.index, columns=self._classes)


class _LeakyClient(_HonestSystem):
    """Posts the training table, a test table carrying the target, and the dataset name."""

    ORACLE: dict = {}

    def _predict_proba(self, X):
        import requests

        def b64(frame: pd.DataFrame) -> str:
            return base64.b64encode(frame.to_csv(index=False).encode()).decode()

        test = X.copy()
        test["target"] = [self.ORACLE.get(_LookupSystem.key(r)) for r in X.select_dtypes("number").to_numpy()]
        body = {"train": b64(pd.DataFrame({"f0": range(300), "target": 0})), "test": b64(test)}
        body |= {"dataset": "my-secret-dataset", "fold": 0}
        requests.post("https://api.example.invalid/v1/predict", json=body, headers={"Authorization": "x"}, timeout=5)
        return super()._predict_proba(X)


def _split(problem_type: str, seed: int = 0):
    data = make_synthetic_split(problem_type, seed=seed, n_samples=600)
    data.identifiers = ("my-secret-dataset",)
    return data


def _oracle(data) -> dict:
    X = pd.concat([data.X_train, data.X_test]).select_dtypes("number")
    y = pd.concat([data.y_train, data.y_test])
    return {_LookupSystem.key(row): str(label) for row, label in zip(X.to_numpy(), y, strict=True)}


def _by_check(results, check: str) -> list:
    return [r for r in results if r.check == check]


def test_honest_system_passes():
    datasets = [_split("binary"), _split("multiclass"), _split("regression")]
    results = audit_system(
        _HonestSystem,
        datasets=datasets,
        checks=("egress", "transduction", "labels", "shuffled", "jitter", "time"),
        time_limit=60,
        log=lambda _msg: None,
    )
    failed = [r for r in results if r.verdict == FAIL]
    assert not failed, failed
    assert all(r.verdict == PASS for r in _by_check(results, "transduction"))
    assert all(r.verdict == PASS for r in _by_check(results, "labels"))


def test_transductive_system_fails_transduction():
    results = audit_system(
        _TransductiveSystem,
        datasets=[_split("binary")],
        checks=("transduction",),
        log=lambda _msg: None,
    )
    (check,) = _by_check(results, "transduction")
    assert check.verdict == FAIL
    assert check.details["verdicts"]["decoys"] == FAIL


def test_label_lookup_is_caught():
    data = _split("binary")
    _LookupSystem.ORACLE = _oracle(data)
    results = audit_system(
        _LookupSystem,
        datasets=[data],
        checks=("labels", "shuffled", "jitter"),
        log=lambda _msg: None,
    )
    verdicts = {r.check: r.verdict for r in results if r.check in ("labels", "shuffled", "jitter")}
    assert verdicts == {"labels": FAIL, "shuffled": FAIL, "jitter": FAIL}


def test_offline_egress_records_the_payload_and_sends_nothing():
    pytest.importorskip("requests")
    data = _split("binary")
    _LeakyClient.ORACLE = _oracle(data)
    results = audit_system(_LeakyClient, datasets=[data], checks=("egress",), offline=True, log=lambda _msg: None)
    (check,) = _by_check(results, "egress")
    assert check.verdict == FAIL
    assert "target column 'target'" in check.summary
    assert "my-secret-dataset" in check.summary
    assert "['fold']" in check.summary or "fold" in check.summary
    (request,) = check.details["requests"]
    assert request["blocked"] is True
    assert request["headers"]["Authorization"] == "<redacted>"
    assert {t["n_rows"] for t in request["tables"]} == {300, len(data.X_test)}


def test_egress_recorder_restores_the_patches():
    import socket

    requests = pytest.importorskip("requests")
    original = (socket.getaddrinfo, requests.Session.send)
    with EgressRecorder(block=True):
        assert socket.getaddrinfo is not original[0]
    assert (socket.getaddrinfo, requests.Session.send) == original


def test_scan_source_flags_dataset_access(tmp_path, monkeypatch):
    package = tmp_path / "fake_api_system"
    package.mkdir()
    (package / "__init__.py").write_text("")
    (package / "system.py").write_text(
        textwrap.dedent(
            '''
            from __future__ import annotations

            import os


            class FakeSystem:
                """Uses 8-fold bagging (a docstring mention is not a hit)."""

                def _fit_system(self, X, y, *, random_state, time_limit):
                    import openml

                    fold = os.environ["FOLD"]
                    return openml.tasks.get_task(1), fold, time_limit
            ''',
        ),
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    module = importlib.import_module("fake_api_system.system")
    try:
        result = scan_source(module.FakeSystem)
    finally:
        sys.modules.pop("fake_api_system.system", None)
        sys.modules.pop("fake_api_system", None)
    categories = {hit["category"] for hit in result.details["hits"]}
    assert result.verdict == WARN
    assert {"dataset or cache access", "task identity"} <= categories
    assert not any("8-fold" in hit["code"] for hit in result.details["hits"])
    assert "random_state" in result.details["ignored_fit_inputs"]
    assert "time_limit" not in result.details["ignored_fit_inputs"]


def test_compare_submitted_results_flags_implausible_gains():
    hosted = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "fold": [0, 0, 0, 0],
            "method": ["M1", "M2", "M1", "M2"],
            "metric_error": [0.20, 0.25, 0.10, 0.12],
        },
    )
    submitted = pd.DataFrame({"dataset": ["a", "b"], "fold": [0, 0], "metric_error": [0.21, 0.005]})
    results = compare_submitted_results(submitted, hosted)
    flagged = [r for r in results if r.verdict == FAIL]
    assert [r.dataset for r in flagged] == ["b"]
    assert results[-1].details["per_split"][0]["rank"] == 2
