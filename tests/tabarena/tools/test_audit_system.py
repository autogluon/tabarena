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
    SKIP,
    WARN,
    EgressRecorder,
    audit_system,
    compare_reproduced_results,
    compare_submitted_results,
    make_decoys,
    make_synthetic_split,
    results_frame,
    scan_source,
    training_copies,
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


class _ProbeRefusingSystem(_HonestSystem):
    """Serves the real training labels and answers any other labels with an outage."""

    ORIGINAL: pd.Series | None = None

    def _fit_system(self, X, y, **kwargs):
        if not y.reset_index(drop=True).equals(self.ORIGINAL.reset_index(drop=True)):
            raise RuntimeError("HTTP 503: model_unavailable")
        return super()._fit_system(X, y, **kwargs)


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


def _per_split(errors: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"dataset": [f"d{i}" for i in range(len(errors))], "fold": 0, "metric_error": errors})


@pytest.mark.parametrize(
    ("ours", "verdict"),
    [
        ([0.20, 0.30, 0.40, 0.50], PASS),  # matches: a seeded system
        ([0.20002, 0.30, 0.40, 0.50], PASS),  # GPU float noise still matches
        ([0.201, 0.299, 0.402, 0.499], PASS),  # run-to-run noise
        ([0.24, 0.30, 0.40, 0.50], PASS),  # one split 1.2x: inside the band, median 1
        ([0.30, 0.30, 0.40, 0.50], WARN),  # one split outside the band
        ([0.23, 0.345, 0.46, 0.575], FAIL),  # 15% worse everywhere: the submitted numbers do not reproduce
    ],
)
def test_compare_reproduced_results(ours, verdict):
    submitted = _per_split([0.20, 0.30, 0.40, 0.50])
    result = compare_reproduced_results(_per_split(ours), submitted)
    assert result.verdict == verdict, result.summary
    assert len(result.details["per_split"]) == 4


def test_compare_reproduced_results_matches_on_dataset_and_split():
    ours = pd.DataFrame({"dataset": ["d0", "d0"], "fold": [0, 1], "metric_error": [0.2, 0.9]})
    submitted = pd.DataFrame({"dataset": ["d0", "d1"], "fold": [0, 0], "metric_error": [0.2, 0.9]})
    result = compare_reproduced_results(ours, submitted)
    assert result.verdict == PASS
    assert [r["fold"] for r in result.details["per_split"]] == [0]
    assert compare_reproduced_results(ours.iloc[1:], submitted).verdict == SKIP


def test_results_frame_uses_the_split_index():
    results = [
        {"framework": "Sys_c1_default", "metric_error": 0.3, "task_metadata": {"name": "d0", "split_idx": 7}},
    ]
    frame = results_frame(results)
    assert frame.to_dict(orient="records") == [
        {"method": "Sys_c1_default", "dataset": "d0", "fold": 7, "metric_error": 0.3},
    ]


def test_a_refused_probe_is_named():
    data = _split("binary")
    _ProbeRefusingSystem.ORIGINAL = data.y_train
    results = audit_system(
        _ProbeRefusingSystem,
        datasets=[data],
        checks=("transduction", "labels", "time"),
        log=lambda _msg: None,
    )
    (probe,) = _by_check(results, "probe")
    assert probe.verdict == FAIL
    assert "on the 'labels' call, after 5 served call(s)" in probe.summary
    assert [c["probe"] for c in probe.details["calls"] if "error" in c] == ["labels"]
    (timing,) = _by_check(results, "time")
    assert timing.summary.startswith("5 calls")


def test_decoys_keep_integer_columns_in_their_type():
    X = pd.DataFrame(
        {
            "count": np.arange(40, dtype=np.uint8),
            "signed": np.arange(-20, 20, dtype=np.int64),
            "real": np.linspace(0, 1, 40),
        },
    )
    decoys = make_decoys(X.iloc[:10], X, seed=0)
    assert dict(decoys.dtypes) == dict(X.dtypes)
    shift = decoys["count"].astype(int).to_numpy() - X["count"].iloc[:10].sample(frac=1.0, random_state=0).to_numpy()
    assert (shift == round(10 * X["count"].std())).all()  # unsigned: shifted up, never below zero
    assert (decoys["real"].sub(X["real"].mean()).abs() > 5 * X["real"].std()).mean() > 0.5


class _BelowChanceSystem(_HonestSystem):
    """Given any labels but the real ones, ranks the test rows backwards: far below chance, keeping no skill."""

    def _fit_system(self, X, y, **kwargs):
        original = _ProbeRefusingSystem.ORIGINAL
        self._flip = not y.reset_index(drop=True).equals(original.reset_index(drop=True))
        return super()._fit_system(X, original if self._flip else y, **kwargs)

    def _predict_proba(self, X):
        proba = super()._predict_proba(X)
        return proba.iloc[:, ::-1].set_axis(proba.columns, axis=1) if self._flip else proba


def test_shuffled_check_judges_the_skill_kept_not_the_distance_from_chance():
    data = _split("binary")
    _ProbeRefusingSystem.ORIGINAL = data.y_train
    results = audit_system(_BelowChanceSystem, datasets=[data], checks=("shuffled",), log=lambda _msg: None)
    (check,) = _by_check(results, "shuffled")
    assert check.details["score"] < 0.2  # far below chance, where a strong honest learner can land
    assert check.verdict == PASS


class _MemorizingSystem(_HonestSystem):
    """Returns the training target of a test row's exact training copy, the model's prediction otherwise."""

    def _fit_system(self, X, y, **kwargs):
        self._memory = dict(zip(map(tuple, X.astype(str).to_numpy()), y, strict=True))
        return super()._fit_system(X, y, **kwargs)

    def _predict(self, X):
        pred = super()._predict(X)
        keys = map(tuple, X.astype(str).to_numpy())
        return pd.Series([self._memory.get(k, p) for k, p in zip(keys, pred, strict=True)], index=X.index)


def _with_training_copies(data, share: float = 0.5):
    """Replace a share of the test rows with copies of training rows (and their targets)."""
    n = int(len(data.X_test) * share)
    data.X_test = pd.concat([data.X_train.iloc[:n], data.X_test.iloc[n:]], ignore_index=True)
    data.y_test = pd.concat([data.y_train.iloc[:n], data.y_test.iloc[n:]], ignore_index=True)
    return data


def test_training_copies():
    X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": ["x", "y", "z"]})
    X_test = pd.DataFrame({"a": [2.0, 2.0, 4.0], "b": ["y", "x", "z"]})
    assert training_copies(X_test, X_train).tolist() == [True, False, False]


def test_jitter_is_judged_on_rows_without_a_training_copy():
    data = _with_training_copies(_split("regression"))
    results = audit_system(_MemorizingSystem, datasets=[data], checks=("jitter",), log=lambda _msg: None)
    (check,) = _by_check(results, "jitter")
    groups = check.details["groups"]
    assert check.details["judged_on"] == "no_copy"
    assert groups["copy"]["system"]["change"] > 1.0  # the copies are lost under the noise: a sharp rise there
    assert groups["all"]["system"]["change"] > 0.25  # judged on all rows, the memorizer would have been flagged
    assert check.verdict == PASS, check.summary
