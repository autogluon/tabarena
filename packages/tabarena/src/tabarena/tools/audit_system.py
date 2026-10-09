"""Audit a benchmarked system before it runs: what its client sends, and whether it learns honestly.

Usage::

    python -P -m tabarena.tools.audit_system --system NAME [--config JSON] [--datasets NAME ...]
        [--checks CHECK ...] [--time-limit S] [--offline] [--seed N] [--json PATH]
    python -P -m tabarena.tools.audit_system --submitted-results per_split.csv [...] [--json PATH]

A system receives the training table *and* the test features (the hosted APIs in one call), and the
TabArena datasets are public, so a system could look the test labels up instead of predicting them,
or adapt to the test batch it is handed. TabArena cannot read the code behind a closed API; this tool
checks the client it can read and probes the behaviour it cannot. The checks:

``source``
    Static scan of the system's package for code a client has no business running: dataset or cache
    access (``openml``, the TabArena caches, file reads), task identity (``task_id``, ``fold``,
    ``y_test``), subprocesses, dynamic code, and the fit inputs the wrapper ignores (``random_state``,
    ``time_limit``, ...). Hits are listed for a reviewer to read.
``egress``
    Records every outbound request of the fit and predict (``requests`` in full, any other library by
    the host it resolves): hosts, header names (secrets redacted), the JSON outline and the tables
    embedded in it (base64 parquet or CSV, record lists). Fails when a test-sized table carries the
    target column or a request goes to a public dataset host (OpenML); warns on extra columns, task
    identifiers (dataset name, task id, fold keys) and hosts beyond one endpoint. ``--offline`` blocks
    the network and aborts the fit at its first request, so the client can be read without a key and
    without sending any data.
``transduction``
    The prediction for a test row must not depend on which other rows are in the batch. Predicts the
    first half of the test rows alone, the rows in reverse order, and the first half next to as many
    decoy rows (numeric features shifted by ten standard deviations, integer columns kept integer and
    inside their type's range, unseen categories), and compares
    each with the full-batch prediction against the run-to-run noise of a second, identical run.
``labels``
    Relabels the training classes with a derangement (regression: negates the target). An honest
    learner's predictions follow the new labels; predictions that still match the original test
    labels came from somewhere other than the training labels.
``shuffled``
    Trains on labels shuffled across rows. An honest learner loses its skill, but what is left of it
    depends on the one permutation drawn: on an easy task a strong learner lands anywhere from an AUC
    of 0.2 to 0.8. The check therefore compares the skill kept above chance (AUC 0.5, the majority
    rate, R^2 0) with the skill on the real labels; a system that keeps most of it is not learning
    from the labels it was given.
``jitter``
    Adds noise of 1% of a feature's standard deviation to the numeric test features. An exact-match
    lookup of public rows breaks while a model degrades smoothly; the degradation is compared with a
    local reference model (histogram gradient boosting) on the same noise. Test rows with an exact
    copy among the training rows are scored apart: a model that leans on such a copy (an in-context
    learner, a nearest-neighbour model) loses it under the noise and may degrade sharply there
    without cheating, so the verdict is judged on the rows without a training copy, where a lookup of
    the public test labels is the only way to an exact match. Too few such rows (or one class only)
    and it falls back to all rows.
``time``
    Wall-clock of every call against the ``time_limit`` the system was given.

``--submitted-results`` compares a submission's per-split errors (a CSV with ``dataset``, ``fold``,
``metric_error``, e.g. the ``per_split.csv`` of a TabArena-Lite run) with every hosted method on the
same splits and flags datasets where it beats the best hosted method by a wide margin. No system is
called for this check.

``compare_reproduced_results`` (no CLI flag; the hosted-API run template calls it from ``smoke`` and
``eval``) is the other direction: it compares our own run of a config with the errors the submitter
reported for it on the same splits. A seeded system reproduces them up to float noise; a gap where our errors
are systematically higher means the self-reported numbers came from something other than what the
API serves us. ``results_frame`` / ``load_results_frame`` turn a run's own ``results.pkl`` files into
the frame it takes.

The probes call the system through its exec model exactly as a benchmark item does (``fit_custom``,
then extra ``predict`` / ``predict_proba`` calls on the fitted object), so a hosted API is called
once per probe and dataset: about eight calls per dataset. The default datasets are three small
TabArena tasks (one per problem type) plus a synthetic binary task drawn from a fresh seed, which
has no public copy. Data is loaded before the egress recorder starts, so only the system's own
traffic is recorded.

Verdicts: ``PASS``, ``WARN`` (read the details), ``FAIL`` (blatant: stop and ask the submitter),
``INFO`` and ``SKIP``. The thresholds catch blatant cheating, not subtle cheating: a provider who
knows these probes can special-case them, so a passing audit is necessary, not sufficient. Exit code
1 when any check failed, 2 on a usage error, 0 otherwise.

Run it with ``python -P`` from the repository root (see :mod:`tabarena.tools.audit_warmup`).
"""

from __future__ import annotations

import argparse
import ast
import base64
import binascii
import contextlib
import inspect
import io
import json
import math
import re
import socket
import sys
import time
import tokenize
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlsplit

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

PASS, WARN, FAIL, INFO, SKIP = "PASS", "WARN", "FAIL", "INFO", "SKIP"
CHECKS = ("source", "egress", "transduction", "labels", "shuffled", "jitter", "time")
DEFAULT_DATASETS = (
    "blood-transfusion-service-center",
    "maternal_health_risk",
    "QSAR_fish_toxicity",
    "synthetic-binary",
)
SYNTHETIC_PREFIX = "synthetic-"
#: The TabArena-v0.1 metric per problem type; the probes score with these whatever the task declares.
DEFAULT_METRICS = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
#: Hosts serving the public datasets the benchmark is built from; a system contacting one fails.
PUBLIC_DATA_HOSTS = ("openml.org",)
#: Hosts that are not a dataset source but that a client has no obvious reason to contact either.
SUSPICIOUS_HOSTS = ("kaggle.com", "archive.ics.uci.edu", "huggingface.co", "github.com", "githubusercontent.com")
_LOCAL_HOSTS = ("localhost", "127.0.0.1", "::1", "0.0.0.0")  # noqa: S104
_SENSITIVE_HEADER = re.compile(r"auth|key|token|secret|cookie|password", re.IGNORECASE)
_UNSEEN_CATEGORY = "__tabarena_audit_unseen__"

#: Source patterns worth a reviewer's eye in a system's client code, by category.
SOURCE_PATTERNS: dict[str, tuple[str, ...]] = {
    "dataset or cache access": (
        r"\bopenml\b",
        r"tabarena_tasks|tabarena_metadata_cache|TABARENA_CACHE",
        r"\.cache[/\\](openml|tabarena)",
        r"get_train_test_split|get_split_indices",
        r"hf_hub_download|snapshot_download|load_dataset\(",
        r"\bkaggle\b",
    ),
    "file access": (
        r"\bopen\(",
        r"read_(csv|parquet|pickle|feather)\(",
        r"pickle\.load|joblib\.load",
        r"\bglob\(|os\.walk|os\.listdir|\.rglob\(|\.iterdir\(",
    ),
    "processes and sockets": (r"\bsubprocess\b|os\.system|os\.popen|\bsocket\.",),
    "dynamic code": (r"\beval\(|\bexec\(|__import__\(|importlib\.import_module",),
    "global patching": (r"\bbuiltins\b|sys\.modules\[|setattr\((np|pd|sklearn|torch|requests)\b",),
    "network clients": (r"\brequests\b|\bhttpx\b|\baiohttp\b|urllib\.request|http\.client|websocket",),
    "environment": (r"os\.environ|getenv\(",),
}
#: Code names that only make sense in a client that knows which task and split it is running on.
TASK_IDENTITY_NAMES = frozenset({"task_id", "tid", "split_idx", "y_test", "dataset_name", "fold", "repeat"})
#: Categories whose hits make the source check WARN rather than INFO.
SOURCE_WARN_CATEGORIES = ("dataset or cache access", "task identity", "global patching")
#: ``compare_reproduced_results``: relative tolerance of matching errors (GPU float noise of a seeded fit is
#: about 1e-5), the median ratio (ours / submitted) that fails, its distance from 1 that warns, and the
#: per-split ratio band outside which a split warns.
REPRODUCTION_RTOL = 1e-4
REPRODUCTION_MEDIAN_FAIL = 1.10
REPRODUCTION_MEDIAN_WARN = 0.02
REPRODUCTION_SPLIT_BAND = (0.8, 1.25)
#: ``shuffled``: the share of the real-label skill above chance kept with shuffled labels that fails / warns,
#: and the least distance from chance (AUC, accuracy or R^2) either verdict needs.
SHUFFLED_RETAINED_FAIL = 0.8
SHUFFLED_RETAINED_WARN = 0.5
SHUFFLED_MIN_MARGIN = 0.1
#: ``jitter``: the fewest test rows a group (with / without a training copy) needs to be scored on its own.
JITTER_MIN_GROUP_ROWS = 30
#: Fit inputs every system receives; a wrapper that never reads one ignores it.
FIT_INPUTS = ("random_state", "time_limit", "num_cpus", "num_gpus", "memory_limit", "eval_metric")


@dataclass
class CheckResult:
    """One check's outcome on one dataset (``dataset`` is ``"-"`` for dataset-free checks)."""

    check: str
    dataset: str
    verdict: str
    summary: str
    details: dict = field(default_factory=dict)


@dataclass
class ProbeData:
    """One train/test split the probes run on, plus what must never leave the process about it."""

    name: str
    problem_type: str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    target_name: str
    validation_metadata: Any = None
    identifiers: tuple[str, ...] = ()
    """Strings identifying the task (dataset name, task ids); a client must not send them."""

    @property
    def eval_metric(self) -> str:
        return DEFAULT_METRICS[self.problem_type]


# --------------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------------
def _subsample(X: pd.DataFrame, y: pd.Series, n: int | None, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if n is None or len(X) <= n:
        return X, y
    idx = X.sample(n=n, random_state=seed).index
    return X.loc[idx], y.loc[idx]


def load_tabarena_split(
    dataset: str,
    *,
    fold: int = 0,
    repeat: int = 0,
    max_train_rows: int | None = 2000,
    max_test_rows: int | None = 1000,
    seed: int = 0,
) -> ProbeData:
    """Load one TabArena-v0.1 split by dataset name (from the local OpenML cache when materialized)."""
    from tabarena.benchmark.task.spec import task_spec_from_task_id_str
    from tabarena.contexts import TabArenaContext

    meta = TabArenaContext().task_metadata
    rows = meta[(meta["dataset"] == dataset) | (meta["tabarena_task_name"] == dataset)]
    if rows.empty:
        raise ValueError(f"Dataset {dataset!r} is not in the TabArena-v0.1 task metadata.")
    row = rows.iloc[0]
    task = task_spec_from_task_id_str(row["task_id_str"]).load()
    X_train, y_train, X_test, y_test = task.get_train_test_split(fold=fold, repeat=repeat)
    X_train, y_train = _subsample(X_train, y_train, max_train_rows, seed)
    X_test, y_test = _subsample(X_test, y_test, max_test_rows, seed)
    validation_metadata = task.get_validation_metadata()
    target_name = validation_metadata.get_target_name()
    identifiers = tuple(
        str(v) for v in {row["dataset"], row["tabarena_task_name"], row["dataset_name"], row["tid"], row["task_id_str"]}
    )
    return ProbeData(
        name=dataset,
        problem_type=str(row["problem_type"]),
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        target_name=target_name,
        validation_metadata=validation_metadata,
        identifiers=identifiers,
    )


def make_synthetic_split(problem_type: str, *, seed: int, n_samples: int = 2000) -> ProbeData:
    """A fresh task with no public copy: sklearn generators, one noise categorical, a 75/25 split."""
    from sklearn.datasets import make_classification, make_friedman1

    rng = np.random.default_rng(seed)
    if problem_type == "regression":
        X, y = make_friedman1(n_samples=n_samples, n_features=8, noise=0.5, random_state=seed)
        y = pd.Series(y, name="target")
    else:
        n_classes = 2 if problem_type == "binary" else 3
        X, y = make_classification(
            n_samples=n_samples,
            n_features=10,
            n_informative=6,
            n_redundant=2,
            n_classes=n_classes,
            flip_y=0.03,
            class_sep=0.8,
            random_state=seed,
        )
        y = pd.Series(np.asarray([f"class_{v}" for v in y]), name="target")
    frame = pd.DataFrame(X, columns=[f"f{i}" for i in range(X.shape[1])])
    frame["cat"] = pd.Series(rng.choice(["a", "b", "c", "d"], size=n_samples), dtype="category")
    test_mask = rng.random(n_samples) < 0.25
    return ProbeData(
        name=f"{SYNTHETIC_PREFIX}{problem_type}",
        problem_type=problem_type,
        X_train=frame[~test_mask],
        y_train=y[~test_mask],
        X_test=frame[test_mask],
        y_test=y[test_mask],
        target_name="target",
        validation_metadata={"target_name": "target"},
        identifiers=(),
    )


def load_probe_data(dataset: str, *, seed: int, **kwargs) -> ProbeData:
    """``synthetic-<problem_type>`` or a TabArena-v0.1 dataset name."""
    if dataset.startswith(SYNTHETIC_PREFIX):
        return make_synthetic_split(dataset.removeprefix(SYNTHETIC_PREFIX), seed=seed)
    return load_tabarena_split(dataset, seed=seed, **kwargs)


# --------------------------------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------------------------------
def _proba_matrix(pred: Any, classes: list) -> np.ndarray:
    """Positional ``(n_rows, n_classes)`` array with columns in ``classes`` order (missing ones 0)."""
    if isinstance(pred, pd.DataFrame):
        by_str = {str(c): c for c in pred.columns}
        cols = [
            pred[by_str[str(c)]].to_numpy(dtype=float) if str(c) in by_str else np.zeros(len(pred)) for c in classes
        ]
        return np.column_stack(cols)
    return np.asarray(pred, dtype=float)


def _labels(pred: np.ndarray, classes: list) -> np.ndarray:
    return np.asarray([str(classes[i]) for i in np.argmax(pred, axis=1)])


def _error(problem_type: str, y_true: pd.Series, pred: np.ndarray, classes: list) -> float:
    """TabArena-v0.1 error: ``1 - AUC`` (binary), log loss (multiclass), RMSE (regression)."""
    from sklearn.metrics import log_loss, roc_auc_score

    y = np.asarray([str(v) for v in y_true]) if problem_type != "regression" else np.asarray(y_true, dtype=float)
    if problem_type == "regression":
        return float(np.sqrt(np.mean((y - pred) ** 2)))
    str_classes = [str(c) for c in classes]
    if problem_type == "binary":
        positive = sorted(set(y))[-1]
        return float(1.0 - roc_auc_score(y == positive, pred[:, str_classes.index(positive)]))
    clipped = np.clip(pred, 1e-15, 1.0)
    clipped = clipped / clipped.sum(axis=1, keepdims=True)
    return float(log_loss(y, clipped, labels=str_classes))


def _mean_abs_diff(a: np.ndarray, b: np.ndarray, scale: float) -> float:
    return float(np.mean(np.abs(np.asarray(a, dtype=float) - np.asarray(b, dtype=float))) / scale)


# --------------------------------------------------------------------------------------------------
# Running the system
# --------------------------------------------------------------------------------------------------
@dataclass
class SystemUnderAudit:
    """A system class plus the config, budget and fit inputs every probe constructs it with."""

    system_cls: type
    config: dict = field(default_factory=dict)
    time_limit: float = 600
    num_cpus: int | None = None
    split_seed: int = 0
    calls: list[dict] = field(default_factory=list)
    """``{"probe", "dataset", "rows", "wall_s"}`` per call to the system (plus ``"error"`` when it raised)."""

    def _timed(self, data: ProbeData, rows: int, probe: str, call: Callable[[], Any]) -> Any:
        entry = {"probe": probe, "dataset": data.name, "rows": rows}
        start = time.monotonic()
        try:
            out = call()
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:200]}"
            self.calls.append({**entry, "wall_s": time.monotonic() - start, "error": error})
            raise
        self.calls.append({**entry, "wall_s": time.monotonic() - start})
        return out

    def fit_predict(self, data: ProbeData, X_train, y_train, X_test, *, probe: str) -> tuple[Any, Any]:
        """Construct, ``fit_custom`` and return ``(fitted_model, test_prediction)`` like a benchmark item."""
        from autogluon.core.metrics import get_metric

        model = self.system_cls(
            problem_type=data.problem_type,
            eval_metric=get_metric(data.eval_metric, problem_type=data.problem_type),
            validation_metadata=data.validation_metadata,
            fit_kwargs={"num_cpus": self.num_cpus, "num_gpus": 0, "memory_limit": None, "time_limit": self.time_limit},
            **self.config,
        )
        out = self._timed(
            data,
            len(X_test),
            probe,
            lambda: model.fit_custom(X_train.copy(), y_train.copy(), X_test.copy(), split_seed=self.split_seed),
        )
        pred = out["probabilities"] if data.problem_type != "regression" else out["predictions"]
        return model, pred

    def predict(self, model, data: ProbeData, X: pd.DataFrame, *, probe: str) -> Any:
        """One more prediction from an already fitted system (a hosted API may refit here)."""
        predict = model.predict_proba if data.problem_type != "regression" else model.predict
        return self._timed(data, len(X), probe, lambda: predict(X.copy()))


def resolve_system(name: str) -> tuple[type, dict]:
    """``(system_cls, first manual config)`` of a registered system, by method or display name."""
    from tabarena.systems import get_system_registry

    registry = get_system_registry()
    for info in registry.values():
        md = info.method_metadata
        if name in (md.method, md.display_name, info.config_generator.name):
            configs = info.config_generator.manual_configs or [{}]
            config = {k: v for k, v in configs[0].items() if k not in ("ag_args", "ag_args_ensemble")}
            return info.system_cls, config
    raise ValueError(f"System {name!r} is not registered. Options: {sorted(registry)}")


# --------------------------------------------------------------------------------------------------
# Egress recording
# --------------------------------------------------------------------------------------------------
class AuditBlockedRequestError(BaseException):
    """Raised in place of a network request by an offline recorder.

    A ``BaseException`` so that a client's retry loop (``except Exception``) does not swallow it and
    sleep through its backoff.
    """


@dataclass
class RecordedRequest:
    method: str
    url: str
    host: str
    headers: dict[str, str]
    body_bytes: int
    outline: Any = None
    tables: list[dict] = field(default_factory=list)
    strings: list[str] = field(default_factory=list)
    keys: list[str] = field(default_factory=list)
    numbers: list[float] = field(default_factory=list)
    status: int | None = None
    response_bytes: int | None = None
    response_keys: list[str] = field(default_factory=list)
    wall_s: float | None = None
    blocked: bool = False


def _try_decode_table(text: str) -> tuple[pd.DataFrame, str] | None:
    """A table encoded in a string, with its encoding: base64 parquet, base64 CSV, or plain CSV."""
    raw: bytes | None = None
    with contextlib.suppress(binascii.Error, ValueError):
        raw = base64.b64decode(text, validate=True)
    candidates = [(raw, "base64 ")] if raw is not None else []
    candidates.append((text.encode(), ""))
    for blob, prefix in candidates:
        if blob[:4] == b"PAR1":
            with contextlib.suppress(Exception):
                return pd.read_parquet(io.BytesIO(blob)), f"{prefix}parquet"
        head = blob[:2000]
        if b"\n" in head and b"," in head:
            with contextlib.suppress(Exception):
                frame = pd.read_csv(io.BytesIO(blob))
                if frame.shape[1] > 1 and len(frame) > 1:
                    return frame, f"{prefix}csv"
    return None


def _table_summary(frame: pd.DataFrame, path: str, encoding: str) -> dict:
    return {"path": path, "encoding": encoding, "n_rows": len(frame), "columns": [str(c) for c in frame.columns]}


def _outline_list(obj: list, path: str, rec: RecordedRequest) -> Any:
    if len(obj) > 20 and all(isinstance(v, dict) for v in obj[:20]):
        with contextlib.suppress(Exception):
            frame = pd.DataFrame(obj)
            rec.tables.append(_table_summary(frame, path, "records"))
            return f"<table records {frame.shape[0]}x{frame.shape[1]}>"
    if len(obj) > 20:
        return f"<list len={len(obj)}>"
    return [_outline(v, f"{path}[{i}]", rec) for i, v in enumerate(obj)]


def _outline_str(obj: str, path: str, rec: RecordedRequest) -> str:
    if len(obj) <= 512:
        rec.strings.append(obj)
        return obj
    decoded = _try_decode_table(obj)
    if decoded is None:
        return f"<str len={len(obj)}>"
    frame, encoding = decoded
    rec.tables.append(_table_summary(frame, path, encoding))
    return f"<table {encoding} {frame.shape[0]}x{frame.shape[1]}>"


def _outline(obj: Any, path: str, rec: RecordedRequest) -> Any:
    """Shape of a JSON body with tables replaced by a summary; fills ``rec``'s tables/strings/keys."""
    if isinstance(obj, dict):
        out = {}
        for key, value in obj.items():
            rec.keys.append(str(key))
            out[key] = _outline(value, f"{path}.{key}", rec)
        return out
    if isinstance(obj, list):
        return _outline_list(obj, path, rec)
    if isinstance(obj, str):
        return _outline_str(obj, path, rec)
    if isinstance(obj, int | float) and not isinstance(obj, bool):
        rec.numbers.append(float(obj))
    return obj if obj is None or isinstance(obj, bool | int | float) else f"<{type(obj).__name__}>"


class EgressRecorder:
    """Record (and with ``block=True`` refuse) the outbound requests made inside the context.

    ``requests`` is recorded in full by wrapping ``requests.Session.send``; every other library is
    seen by the host it resolves (``socket.getaddrinfo``). Blocking raises
    :class:`AuditBlockedRequestError` before anything is sent.
    """

    def __init__(self, *, block: bool = False):
        self.block = block
        self.requests: list[RecordedRequest] = []
        self.hosts: list[str] = []
        self._restore: list[Callable[[], None]] = []

    def __enter__(self) -> EgressRecorder:
        original_getaddrinfo = socket.getaddrinfo

        def getaddrinfo(host, *args, **kwargs):
            name = host.decode() if isinstance(host, bytes) else str(host)
            if host is not None and name not in _LOCAL_HOSTS:
                self.hosts.append(name)
                if self.block:
                    raise AuditBlockedRequestError(f"audit_system --offline blocked a connection to {name}")
            return original_getaddrinfo(host, *args, **kwargs)

        socket.getaddrinfo = getaddrinfo
        self._restore.append(lambda: setattr(socket, "getaddrinfo", original_getaddrinfo))
        with contextlib.suppress(ImportError):
            import requests

            original_send = requests.Session.send
            recorder = self

            def send(session, request, **kwargs):
                rec = recorder._record(request)
                if recorder.block:
                    rec.blocked = True
                    raise AuditBlockedRequestError(f"audit_system --offline blocked {request.method} {request.url}")
                start = time.monotonic()
                response = original_send(session, request, **kwargs)
                rec.wall_s = time.monotonic() - start
                rec.status = response.status_code
                with contextlib.suppress(Exception):
                    rec.response_bytes = len(response.content)
                    body = response.json()
                    rec.response_keys = sorted(body) if isinstance(body, dict) else []
                return response

            requests.Session.send = send
            self._restore.append(lambda: setattr(requests.Session, "send", original_send))
        return self

    def __exit__(self, *exc) -> None:
        for restore in reversed(self._restore):
            restore()
        self._restore.clear()

    def _record(self, request) -> RecordedRequest:
        url = str(request.url)
        headers = {
            str(k): ("<redacted>" if _SENSITIVE_HEADER.search(str(k)) else str(v)) for k, v in request.headers.items()
        }
        body = request.body or b""
        body_bytes = body if isinstance(body, bytes) else str(body).encode()
        rec = RecordedRequest(
            method=str(request.method),
            url=url,
            host=urlsplit(url).hostname or "",
            headers=headers,
            body_bytes=len(body_bytes),
        )
        parsed: Any = None
        with contextlib.suppress(ValueError, UnicodeDecodeError):
            parsed = json.loads(body_bytes.decode())
        if parsed is not None:
            rec.outline = _outline(parsed, "$", rec)
        elif body_bytes:
            decoded = _try_decode_table(body_bytes.decode(errors="ignore"))
            if decoded is not None:
                rec.tables.append(_table_summary(decoded[0], "$", decoded[1]))
            rec.outline = f"<{headers.get('Content-Type', 'body')} {len(body_bytes)} bytes>"
        self.requests.append(rec)
        return rec


def _host_matches(host: str, suffixes: Iterable[str]) -> bool:
    return any(host == s or host.endswith(f".{s}") for s in suffixes)


def analyze_egress(data: ProbeData, requests: list[RecordedRequest], hosts: list[str], *, n_train: int) -> CheckResult:
    """Judge what the client sent for ``data`` (see the module docstring, ``egress``)."""
    problems: list[str] = []
    warnings: list[str] = []
    all_hosts = sorted({*hosts, *(r.host for r in requests if r.host)})
    data_hosts = [h for h in all_hosts if _host_matches(h, PUBLIC_DATA_HOSTS)]
    if data_hosts:
        problems.append(f"contacted public dataset host(s) {data_hosts}")
    odd_hosts = [h for h in all_hosts if _host_matches(h, SUSPICIOUS_HOSTS)]
    if odd_hosts:
        warnings.append(f"contacted {odd_hosts}")
    api_hosts = [h for h in all_hosts if h not in data_hosts and h not in odd_hosts]
    if len(api_hosts) > 1:
        warnings.append(f"more than one endpoint host {api_hosts}")

    train_cols = {str(c) for c in data.X_train.columns}
    for rec in requests:
        for table in rec.tables:
            cols = set(table["columns"])
            looks_train = table["n_rows"] == n_train and data.target_name in cols
            if looks_train:
                extra = sorted(cols - train_cols - {data.target_name})
                if extra:
                    warnings.append(f"training table {table['path']} carries extra columns {extra}")
                continue
            if data.target_name in cols:
                problems.append(
                    f"table {table['path']} ({table['n_rows']} rows) carries the target column {data.target_name!r} "
                    "and is not the training table",
                )
            extra = sorted(cols - train_cols - {data.target_name})
            if extra:
                warnings.append(f"table {table['path']} carries columns not in the features: {extra}")
        sent = [s.lower() for s in rec.strings] + [k.lower() for k in rec.keys]
        for ident in data.identifiers:
            ident_l = ident.lower()
            if ident_l.isdigit():
                if any(s == ident_l for s in sent) or any(n == float(ident_l) for n in rec.numbers):
                    warnings.append(f"sent the task identifier {ident}")
            elif len(ident_l) >= 5 and any(ident_l in s for s in sent):
                warnings.append(f"sent the dataset name {ident!r}")
        id_keys = sorted(
            {k for k in rec.keys if re.fullmatch(r"(?i)(fold|repeat|split|split_idx|task_?id|tid|dataset)", k)}
        )
        if id_keys:
            warnings.append(f"request keys name the split or task: {id_keys}")

    verdict = FAIL if problems else WARN if warnings else INFO
    shown = [
        {
            "method": r.method,
            "url": r.url,
            "headers": r.headers,
            "body_bytes": r.body_bytes,
            "outline": r.outline,
            "tables": r.tables,
            "status": r.status,
            "response_keys": r.response_keys,
            "blocked": r.blocked,
        }
        for r in requests
    ]
    summary = "; ".join(problems + warnings) or (
        f"{len(requests)} request(s) to {api_hosts or ['(no HTTP host)']}, tables "
        f"{[(t['n_rows'], len(t['columns'])) for r in requests for t in r.tables]}"
    )
    return CheckResult("egress", data.name, verdict, summary, {"hosts": all_hosts, "requests": shown})


# --------------------------------------------------------------------------------------------------
# Static source scan
# --------------------------------------------------------------------------------------------------
def _docstring_lines(source: str) -> set[int]:
    """Line numbers covered by module, class and function docstrings."""
    lines: set[int] = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) and node.body:
            first = node.body[0]
            if (
                isinstance(first, ast.Expr)
                and isinstance(first.value, ast.Constant)
                and isinstance(first.value.value, str)
            ):
                lines.update(range(first.lineno, (first.end_lineno or first.lineno) + 1))
    return lines


def _identity_name_lines(source: str) -> set[int]:
    """Lines whose code (not strings or comments) uses one of :data:`TASK_IDENTITY_NAMES`."""
    lines: set[int] = set()
    with contextlib.suppress(tokenize.TokenError):
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type == tokenize.NAME and token.string in TASK_IDENTITY_NAMES:
                lines.add(token.start[0])
    return lines


def scan_source(system_cls: type) -> CheckResult:
    """List the code lines of the system's package that match :data:`SOURCE_PATTERNS`.

    Docstrings and comments are skipped; task identity is matched on code names only.
    """
    source_file = Path(inspect.getsourcefile(system_cls))
    package_dir = source_file.parent
    files = sorted(p for p in package_dir.rglob("*.py") if "tests" not in p.relative_to(package_dir).parts)
    hits: list[dict] = []
    for path in files:
        source = path.read_text()
        skip = _docstring_lines(source)
        identity = _identity_name_lines(source)
        for lineno, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if not stripped or stripped.startswith("#") or lineno in skip:
                continue
            code = line.split(" #", 1)[0]
            category = "task identity" if lineno in identity else None
            for name, patterns in SOURCE_PATTERNS.items():
                if category is None and any(re.search(p, code) for p in patterns):
                    category = name
            if category is not None:
                hits.append({"file": str(path), "line": lineno, "category": category, "code": stripped[:160]})
    class_source = inspect.getsource(system_cls)
    class_code = "\n".join(
        line for i, line in enumerate(class_source.splitlines(), start=1) if i not in _docstring_lines(class_source)
    )
    ignored = [name for name in FIT_INPUTS if len(re.findall(rf"\b{name}\b", class_code)) <= 1]
    by_category: dict[str, int] = {}
    for hit in hits:
        by_category[hit["category"]] = by_category.get(hit["category"], 0) + 1
    warn = any(c in SOURCE_WARN_CATEGORIES for c in by_category)
    summary = ", ".join(f"{c}: {n}" for c, n in sorted(by_category.items())) or "no matches"
    if ignored:
        summary += f"; fit inputs never read: {ignored}"
    return CheckResult(
        "source",
        "-",
        WARN if warn else INFO,
        summary,
        {"package": str(package_dir), "files": [str(f) for f in files], "hits": hits, "ignored_fit_inputs": ignored},
    )


# --------------------------------------------------------------------------------------------------
# Behaviour probes
# --------------------------------------------------------------------------------------------------
def _numeric_columns(X: pd.DataFrame) -> list:
    return [
        c
        for c in X.columns
        if pd.api.types.is_numeric_dtype(X[c]) and not pd.api.types.is_bool_dtype(X[c]) and X[c].nunique() > 2
    ]


def make_decoys(X: pd.DataFrame, X_train: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Rows far from the training distribution: numerics shifted by 10 std, unseen categories."""
    rng = np.random.default_rng(seed)
    decoys = X.sample(frac=1.0, random_state=seed).copy()
    numeric = set(_numeric_columns(X_train))
    for col in decoys.columns:
        if col in numeric:
            std = float(X_train[col].std()) or 1.0
            signs = rng.choice([-1.0, 1.0], size=len(decoys))
            dtype = X_train[col].dtype
            if pd.api.types.is_integer_dtype(dtype):
                # A whole shift that keeps the column's type: a server may cast the test rows to the training schema.
                bounds = np.iinfo(dtype)
                if bounds.min == 0:
                    signs = np.ones(len(decoys))
                shifted = decoys[col].astype(float) + max(1.0, round(10.0 * std)) * signs
                decoys[col] = shifted.clip(bounds.min, bounds.max).astype(dtype)
            else:
                decoys[col] = decoys[col].astype(float) + 10.0 * std * signs
        elif isinstance(decoys[col].dtype, pd.CategoricalDtype):
            categories = [*decoys[col].cat.categories, _UNSEEN_CATEGORY]
            decoys[col] = pd.Categorical([_UNSEEN_CATEGORY] * len(decoys), categories=categories)
        elif decoys[col].dtype == object:
            decoys[col] = _UNSEEN_CATEGORY
    decoys.index = [f"decoy_{i}" for i in range(len(decoys))]
    return decoys


def jitter(X: pd.DataFrame, X_train: pd.DataFrame, *, scale: float, seed: int) -> pd.DataFrame:
    """``X`` with Gaussian noise of ``scale`` training standard deviations on the numeric columns."""
    rng = np.random.default_rng(seed)
    out = X.copy()
    for col in _numeric_columns(X_train):
        std = float(X_train[col].std()) or 1.0
        out[col] = out[col].astype(float) + rng.normal(0.0, scale * std, size=len(out))
    return out


def _derangement(classes: list) -> dict:
    """Map each class to the next one (a cyclic shift: no class keeps its label)."""
    return {c: classes[(i + 1) % len(classes)] for i, c in enumerate(classes)}


def _relabel(y: pd.Series, mapping: dict) -> pd.Series:
    out = y.astype(object).map(mapping)
    return out.astype(y.dtype) if isinstance(y.dtype, pd.CategoricalDtype) else out


#: Threads for the local reference model: OpenMP over every core of a large head node is far slower.
REFERENCE_THREADS = 4


def _fit_reference(data: ProbeData) -> Callable[[pd.DataFrame], np.ndarray]:
    """A local histogram-gradient-boosting reference; returns its predict function (proba or values)."""
    from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
    from sklearn.preprocessing import OrdinalEncoder
    from threadpoolctl import threadpool_limits

    numeric = [c for c in data.X_train.columns if pd.api.types.is_numeric_dtype(data.X_train[c])]
    other = [c for c in data.X_train.columns if c not in numeric]
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-2)
    if other:
        encoder.fit(data.X_train[other].astype(str))

    def encode(X: pd.DataFrame) -> np.ndarray:
        parts = [X[numeric].astype(float).to_numpy()] if numeric else []
        if other:
            parts.append(encoder.transform(X[other].astype(str)))
        return np.hstack(parts)

    with threadpool_limits(REFERENCE_THREADS):
        if data.problem_type == "regression":
            model = HistGradientBoostingRegressor(random_state=0).fit(encode(data.X_train), data.y_train.astype(float))
        else:
            model = HistGradientBoostingClassifier(random_state=0).fit(
                encode(data.X_train), data.y_train.astype(str).to_numpy()
            )

    def predict(X: pd.DataFrame):
        with threadpool_limits(REFERENCE_THREADS):
            if data.problem_type == "regression":
                return model.predict(encode(X))
            return pd.DataFrame(model.predict_proba(encode(X)), columns=list(model.classes_))

    return predict


def _classes(data: ProbeData) -> list:
    return sorted({str(v) for v in data.y_train} | {str(v) for v in data.y_test})


def _verdict_vs_noise(diff: float, tol: float) -> str:
    if diff <= tol:
        return PASS
    return WARN if diff <= 3 * tol else FAIL


def training_copies(X_test: pd.DataFrame, X_train: pd.DataFrame) -> np.ndarray:
    """Whether each test row has an exact copy (every feature equal) among the training rows."""
    columns = list(X_train.columns)
    keys = set(map(tuple, X_train[columns].astype(str).to_numpy()))
    return np.asarray([tuple(row) in keys for row in X_test[columns].astype(str).to_numpy()], dtype=bool)


def _degradation(data: ProbeData, classes: list, base: np.ndarray, jit: np.ndarray, rows: np.ndarray) -> dict | None:
    """Error before and after the noise on ``rows``; ``None`` when they are too few to score."""
    y = data.y_test.to_numpy()[rows]
    if rows.sum() < JITTER_MIN_GROUP_ROWS or (data.problem_type == "binary" and len({str(v) for v in y}) < 2):
        return None
    before = _error(data.problem_type, y, base[rows], classes)
    after = _error(data.problem_type, y, jit[rows], classes)
    return {
        "rows": int(rows.sum()),
        "error": before,
        "jitter_error": after,
        "change": (after - before) / max(before, 1e-6),
    }


def _change(stats: dict) -> str:
    """The relative error change (``from zero`` when there was no error before the noise)."""
    return "from zero" if stats["error"] < 1e-6 else f"{stats['change']:+.1%}"


def _jitter_check(
    data: ProbeData,
    classes: list,
    base: np.ndarray,
    jit: np.ndarray,
    ref_base: np.ndarray,
    ref_jit: np.ndarray,
) -> CheckResult:
    """The jitter verdict, judged on the test rows without a training copy when there are enough of them."""
    copied = training_copies(data.X_test, data.X_train)
    groups = {"all": np.ones(len(copied), dtype=bool), "no_copy": ~copied, "copy": copied}
    stats = {
        name: {
            "system": _degradation(data, classes, base, jit, rows),
            "reference": _degradation(data, classes, ref_base, ref_jit, rows),
        }
        for name, rows in groups.items()
    }
    judged = "no_copy" if stats["no_copy"]["system"] and stats["no_copy"]["reference"] else "all"
    sys_stats, ref_stats = stats[judged]["system"], stats[judged]["reference"]
    r_sys, r_ref = sys_stats["change"], ref_stats["change"]
    bound = max(r_ref, 0.05)
    abs_ok = data.problem_type == "regression" or (sys_stats["jitter_error"] - sys_stats["error"]) > 0.01
    if r_sys > 1.0 and r_sys > 4 * bound and abs_ok:
        verdict = FAIL
    elif r_sys > 0.25 and r_sys > 2 * bound and abs_ok:
        verdict = WARN
    else:
        verdict = PASS
    where = (
        f"the {sys_stats['rows']} test rows without a training copy"
        if judged == "no_copy"
        else f"all {sys_stats['rows']} test rows ({int((~copied).sum())} without a training copy, too few to judge alone)"
    )
    summary = (
        f"1% feature noise on {where}: error {sys_stats['error']:.4f} -> {sys_stats['jitter_error']:.4f} "
        f"({_change(sys_stats)}); reference {ref_stats['error']:.4f} -> {ref_stats['jitter_error']:.4f} ({_change(ref_stats)})"
    )
    copy_sys, copy_ref = stats["copy"]["system"], stats["copy"]["reference"]
    if judged == "no_copy" and copy_sys and copy_ref:
        summary += (
            f"; the {copy_sys['rows']} rows with a training copy: {copy_sys['error']:.4f} -> "
            f"{copy_sys['jitter_error']:.4f} ({_change(copy_sys)}), reference {_change(copy_ref)}"
        )
    return CheckResult(
        "jitter",
        data.name,
        verdict,
        summary,
        {"judged_on": judged, "rows_with_training_copy": int(copied.sum()), "groups": stats},
    )


def probe_dataset(sut: SystemUnderAudit, data: ProbeData, checks: Iterable[str], *, seed: int) -> list[CheckResult]:
    """Run the behaviour probes (and the egress analysis of their requests) on one split."""
    checks = set(checks)
    results: list[CheckResult] = []
    is_reg = data.problem_type == "regression"
    classes = _classes(data)
    scale = (float(data.y_train.astype(float).std()) or 1.0) if is_reg else 1.0

    def as_array(pred) -> np.ndarray:
        if is_reg:
            return np.asarray(pred, dtype=float).reshape(-1)
        return _proba_matrix(pred, classes)

    def err(y_true: pd.Series, pred: np.ndarray) -> float:
        return _error(data.problem_type, y_true, pred, classes)

    with EgressRecorder() as recorder:
        model, base_pred = sut.fit_predict(data, data.X_train, data.y_train, data.X_test, probe="base")
        base = as_array(base_pred)
    if "egress" in checks:
        results.append(analyze_egress(data, recorder.requests, recorder.hosts, n_train=len(data.X_train)))
    base_err = err(data.y_test, base)
    reference = _fit_reference(data)
    ref_base = as_array(reference(data.X_test))
    ref_err = err(data.y_test, ref_base)
    results.append(
        CheckResult(
            "baseline",
            data.name,
            INFO,
            f"error {base_err:.4f} vs local reference {ref_err:.4f} ({data.eval_metric}, {len(data.X_train)} train / "
            f"{len(data.X_test)} test rows)",
            {"error": base_err, "reference_error": ref_err, "ratio_to_reference": base_err / max(ref_err, 1e-12)},
        ),
    )

    if "transduction" in checks:
        _, repeat_pred = sut.fit_predict(data, data.X_train, data.y_train, data.X_test, probe="repeat")
        noise = _mean_abs_diff(base, as_array(repeat_pred), scale)
        tol = max(3 * noise, 1e-3)
        half = np.arange(math.ceil(len(data.X_test) / 2))
        X_half = data.X_test.iloc[half]
        d_half = _mean_abs_diff(base[half], as_array(sut.predict(model, data, X_half, probe="half")), scale)
        reversed_pred = as_array(sut.predict(model, data, data.X_test.iloc[::-1], probe="reversed"))[::-1]
        d_rev = _mean_abs_diff(base, reversed_pred, scale)
        X_decoy = pd.concat([X_half, make_decoys(X_half, data.X_train, seed)])
        decoy_pred = as_array(sut.predict(model, data, X_decoy, probe="decoys"))[: len(half)]
        d_dec = _mean_abs_diff(base[half], decoy_pred, scale)
        variants = {"half": d_half, "reversed": d_rev, "decoys": d_dec}
        verdicts = {k: _verdict_vs_noise(v, tol) for k, v in variants.items()}
        worst = FAIL if FAIL in verdicts.values() else WARN if WARN in verdicts.values() else PASS
        results.append(
            CheckResult(
                "transduction",
                data.name,
                worst,
                f"repeat noise {noise:.2e}; batch-composition change half {d_half:.2e}, reversed {d_rev:.2e}, "
                f"decoys {d_dec:.2e} (tolerance {tol:.2e}{', deterministic' if noise == 0 else ''})",
                {"noise": noise, "tolerance": tol, **variants, "verdicts": verdicts},
            ),
        )

    if "labels" in checks:
        if is_reg:
            _, flip_pred = sut.fit_predict(data, data.X_train, -data.y_train.astype(float), data.X_test, probe="labels")
            flip = as_array(flip_pred)
            y = data.y_test.astype(float).to_numpy()
            corr_base = float(np.corrcoef(base, y)[0, 1])
            corr_follow = float(np.corrcoef(flip, -y)[0, 1])
            corr_orig = float(np.corrcoef(flip, y)[0, 1])
            if corr_orig > 0.3 and corr_orig > corr_follow:
                verdict = FAIL
            elif corr_follow < corr_base - 0.2:
                verdict = WARN
            else:
                verdict = PASS
            summary = (
                f"negated target: corr with -y {corr_follow:.3f} (base corr {corr_base:.3f}), "
                f"corr with the original y {corr_orig:.3f}"
            )
            details = {"corr_base": corr_base, "corr_follow": corr_follow, "corr_original": corr_orig}
        else:
            train_classes = sorted({str(v) for v in data.y_train})
            raw_by_str = {str(v): v for v in data.y_train.unique()}
            mapping = _derangement(train_classes)
            y_perm = _relabel(data.y_train, {raw_by_str[k]: raw_by_str[v] for k, v in mapping.items()})
            _, perm_pred = sut.fit_predict(data, data.X_train, y_perm, data.X_test, probe="labels")
            perm = as_array(perm_pred)
            y_true = np.asarray([str(v) for v in data.y_test])
            y_mapped = np.asarray([mapping.get(v, v) for v in y_true])
            pred_labels = _labels(perm, classes)
            acc_base = float(np.mean(_labels(base, classes) == y_true))
            acc_follow = float(np.mean(pred_labels == y_mapped))
            acc_orig = float(np.mean(pred_labels == y_true))
            majority = float(pd.Series(y_true).value_counts(normalize=True).iloc[0])
            if acc_orig > acc_follow + 0.1 and acc_orig > majority + 0.05:
                verdict = FAIL
            elif acc_follow < acc_base - 0.1:
                verdict = WARN
            else:
                verdict = PASS
            summary = (
                f"relabeled classes {mapping}: accuracy on the new labels {acc_follow:.3f} (base {acc_base:.3f}), "
                f"on the original labels {acc_orig:.3f} (majority rate {majority:.3f})"
            )
            details = {
                "mapping": mapping,
                "acc_base": acc_base,
                "acc_follow": acc_follow,
                "acc_original": acc_orig,
                "majority": majority,
            }
        results.append(CheckResult("labels", data.name, verdict, summary, details))

    if "shuffled" in checks:
        rng = np.random.default_rng(seed)
        y_shuf = pd.Series(rng.permutation(data.y_train.to_numpy()), index=data.y_train.index, name=data.y_train.name)
        if isinstance(data.y_train.dtype, pd.CategoricalDtype):
            y_shuf = y_shuf.astype(data.y_train.dtype)
        _, shuf_pred = sut.fit_predict(data, data.X_train, y_shuf, data.X_test, probe="shuffled")
        shuf = as_array(shuf_pred)
        y_true = np.asarray([str(v) for v in data.y_test]) if not is_reg else data.y_test.astype(float).to_numpy()
        if data.problem_type == "binary":
            score_name, chance = "AUC", 0.5
            score, score_base = 1.0 - err(data.y_test, shuf), 1.0 - base_err
        elif data.problem_type == "multiclass":
            score_name, chance = "accuracy", float(pd.Series(y_true).value_counts(normalize=True).iloc[0])
            score = float(np.mean(_labels(shuf, classes) == y_true))
            score_base = float(np.mean(_labels(base, classes) == y_true))
        else:
            score_name, chance = "R^2", 0.0
            score = float(1.0 - np.mean((y_true - shuf) ** 2) / np.var(y_true))
            score_base = float(1.0 - np.mean((y_true - base) ** 2) / np.var(y_true))
        retained = (score - chance) / max(score_base - chance, 1e-9)
        above = score > chance + SHUFFLED_MIN_MARGIN
        if above and retained > SHUFFLED_RETAINED_FAIL:
            verdict = FAIL
        elif above and retained > SHUFFLED_RETAINED_WARN:
            verdict = WARN
        else:
            verdict = PASS
        summary = (
            f"{score_name} with shuffled training labels {score:.3f} (chance {chance:.3f}, real labels {score_base:.3f}): "
            f"{retained:.0%} of the skill above chance kept"
        )
        details = {"score": score, "score_real_labels": score_base, "chance": chance, "retained": retained}
        results.append(CheckResult("shuffled", data.name, verdict, summary, details))

    if "jitter" in checks:
        if not _numeric_columns(data.X_train):
            results.append(CheckResult("jitter", data.name, SKIP, "no numeric feature to perturb"))
        else:
            X_jit = jitter(data.X_test, data.X_train, scale=0.01, seed=seed)
            jit = as_array(sut.predict(model, data, X_jit, probe="jitter"))
            results.append(
                _jitter_check(data, classes, base, jit, ref_base, as_array(reference(X_jit))),
            )

    with contextlib.suppress(Exception):
        model.cleanup()
    return results


def _probe_failure(sut: SystemUnderAudit, dataset: str, exc: Exception) -> CheckResult:
    """A probe run that raised, naming the call that failed and the calls that succeeded before it.

    An outage fails the first call of a dataset; a system that serves the plain fit but fails on the
    relabeled, shuffled or decoy requests is refusing the probes, which a reviewer must read as such.
    """
    calls = [c for c in sut.calls if c["dataset"] == dataset]
    failed = next((c["probe"] for c in calls if "error" in c), None)
    served = [c["probe"] for c in calls if "error" not in c]
    where = f" on the {failed!r} call" if failed else ""
    after = f"after {len(served)} served call(s) {served}" if served else "on the first call"
    return CheckResult(
        "probe",
        dataset,
        FAIL,
        f"the system raised {type(exc).__name__}{where}, {after}: {str(exc)[:300]}",
        {"calls": calls},
    )


def check_time(sut: SystemUnderAudit) -> CheckResult:
    """Every served call's wall-clock against the ``time_limit`` the system was handed."""
    served = [c for c in sut.calls if "error" not in c]
    if not served:
        return CheckResult("time", "-", SKIP, "no call was served")
    walls = np.asarray([c["wall_s"] for c in served])
    worst = served[int(np.argmax(walls))]
    limit = sut.time_limit
    if walls.max() > 1.5 * limit + 60:
        verdict = FAIL
    elif walls.max() > 1.1 * limit + 30:
        verdict = WARN
    else:
        verdict = PASS
    return CheckResult(
        "time",
        "-",
        verdict,
        f"{len(walls)} calls, median {np.median(walls):.1f}s, max {walls.max():.1f}s "
        f"({worst['probe']} on {worst['dataset']}) against time_limit {limit:g}s",
        {"calls": sut.calls},
    )


def offline_egress(sut: SystemUnderAudit, data: ProbeData) -> CheckResult:
    """Fit with the network blocked: records the first request the client tries, sends nothing."""
    error = None
    with EgressRecorder(block=True) as recorder:
        try:
            sut.fit_predict(data, data.X_train, data.y_train, data.X_test, probe="offline")
        except AuditBlockedRequestError:
            pass
        except Exception as exc:
            error = f"{type(exc).__name__}: {str(exc)[:300]}"
    result = analyze_egress(data, recorder.requests, recorder.hosts, n_train=len(data.X_train))
    if not recorder.requests and not recorder.hosts:
        result.verdict = WARN
        result.summary = f"no outbound request was attempted{f' ({error})' if error else ''}"
    elif error:
        result.details["error"] = error
    result.summary = f"[offline] {result.summary}"
    return result


# --------------------------------------------------------------------------------------------------
# Submitted results versus the hosted methods
# --------------------------------------------------------------------------------------------------
def compare_submitted_results(
    submitted: pd.DataFrame, hosted: pd.DataFrame, *, label: str = "submitted"
) -> list[CheckResult]:
    """Flag datasets where a submission beats every hosted method on the same split by a wide margin.

    ``submitted`` and ``hosted`` carry ``dataset``, ``fold`` and ``metric_error`` (``hosted`` also
    ``method``, and ``imputed`` when present). A ratio to the best hosted error below 0.5 is a WARN,
    below 0.2 a FAIL: label lookup shows up as near-zero errors, a better learner as small gains.
    """
    if "imputed" in hosted:
        hosted = hosted[~hosted["imputed"].astype(bool)]
    rows = []
    for (dataset, fold), group in submitted.groupby(["dataset", "fold"]):
        ref = hosted[(hosted["dataset"] == dataset) & (hosted["fold"] == fold)]
        if ref.empty:
            continue
        err = float(group["metric_error"].iloc[0])
        best = float(ref["metric_error"].min())
        rows.append(
            {
                "dataset": dataset,
                "fold": int(fold),
                "error": err,
                "best_hosted": best,
                "best_method": str(ref.loc[ref["metric_error"].idxmin(), "method"]),
                "ratio": err / best if best > 0 else math.inf if err > 0 else 1.0,
                "rank": int((ref["metric_error"] < err).sum()) + 1,
            },
        )
    if not rows:
        return [CheckResult("submitted", label, SKIP, "no split overlaps the hosted results")]
    frame = pd.DataFrame(rows)
    out = []
    for row in frame.itertuples():
        if row.ratio < 0.5:
            out.append(
                CheckResult(
                    "submitted",
                    row.dataset,
                    FAIL if row.ratio < 0.2 else WARN,
                    f"fold {row.fold}: error {row.error:.4g} is {row.ratio:.2f}x the best hosted "
                    f"({row.best_method}, {row.best_hosted:.4g})",
                    row._asdict(),
                ),
            )
    out.append(
        CheckResult(
            "submitted",
            label,
            INFO,
            f"{len(frame)} splits: first on {int((frame['rank'] == 1).sum())}, median ratio to the best hosted "
            f"{frame['ratio'].median():.3f}, min {frame['ratio'].min():.3f}, median rank {frame['rank'].median():g} "
            f"of {hosted['method'].nunique()}",
            {"per_split": frame.to_dict(orient="records")},
        ),
    )
    return out


def compare_reproduced_results(
    ours: pd.DataFrame, submitted: pd.DataFrame, *, label: str = "reproduced"
) -> CheckResult:
    """Compare a run's errors with the errors a submitter reported for the same config on the same splits.

    Both frames carry ``dataset``, ``fold`` (the split index) and ``metric_error`` for one config, e.g.
    ``results_frame`` of a smoke or of the full run against the submitter's TabArena-Lite ``per_split.csv``.
    A system that seeds its fit from the split reproduces the submitted errors up to float noise
    (``REPRODUCTION_RTOL``, PASS). Otherwise the ratio of our error to the submitted one on each split tells
    how far the two runs are apart: a median
    above ``REPRODUCTION_MEDIAN_FAIL`` is a FAIL (the self-reported numbers are better than what the system
    delivers to us), a median more than ``REPRODUCTION_MEDIAN_WARN`` away from 1 or a split outside
    ``REPRODUCTION_SPLIT_BAND`` is a WARN, and anything else passes as run-to-run noise.
    """
    merged = ours[["dataset", "fold", "metric_error"]].merge(
        submitted[["dataset", "fold", "metric_error"]],
        on=["dataset", "fold"],
        suffixes=("_ours", "_submitted"),
    )
    if merged.empty:
        return CheckResult("reproduced", label, SKIP, "no split of the run is in the submitted results")
    a, b = merged["metric_error_ours"].astype(float), merged["metric_error_submitted"].astype(float)
    merged["match"] = np.isclose(a, b, rtol=REPRODUCTION_RTOL, atol=1e-12)
    merged["ratio"] = [x / y if y > 0 else 1.0 if x == 0 else math.inf for x, y in zip(a, b, strict=True)]
    median = float(merged["ratio"].median())
    low, high = REPRODUCTION_SPLIT_BAND
    outliers = merged[(merged["ratio"] < low) | (merged["ratio"] > high)]
    n, n_same = len(merged), int(merged["match"].sum())
    if n_same == n:
        verdict = PASS
    elif median > REPRODUCTION_MEDIAN_FAIL:
        verdict = FAIL
    elif abs(median - 1) > REPRODUCTION_MEDIAN_WARN or not outliers.empty:
        verdict = WARN
    else:
        verdict = PASS
    summary = (
        f"{n} splits: {n_same} match the submitted errors (rtol {REPRODUCTION_RTOL:g}), median ratio "
        f"ours/submitted {median:.3f}; of the others, ours is worse on {int(((merged['ratio'] > 1) & ~merged['match']).sum())}"
        f" and better on {int(((merged['ratio'] < 1) & ~merged['match']).sum())}"
    )
    if not outliers.empty:
        shown = ", ".join(f"{r.dataset} fold {r.fold} ({r.ratio:.2f}x)" for r in outliers.head(5).itertuples())
        summary += f"; outside [{low:g}, {high:g}]: {shown}{' ...' if len(outliers) > 5 else ''}"
    return CheckResult("reproduced", label, verdict, summary, {"per_split": merged.to_dict(orient="records")})


def results_frame(results: Iterable[dict]) -> pd.DataFrame:
    """``method``, ``dataset``, ``fold`` (the split index) and ``metric_error`` of TabArena result dicts."""
    rows = [
        {
            "method": r["framework"],
            "dataset": r["task_metadata"]["name"],
            "fold": int(r["task_metadata"]["split_idx"]),
            "metric_error": float(r["metric_error"]),
        }
        for r in results
    ]
    return pd.DataFrame(rows, columns=["method", "dataset", "fold", "metric_error"])


def load_results_frame(data_dir: str | Path) -> pd.DataFrame:
    """``results_frame`` of every ``results.pkl`` under a run's own output directory.

    Only for results TabArena wrote itself: unpickling runs code, so a submitter's pickles are never loaded.
    """
    from tabarena.utils.pickle_utils import load_pickle

    return results_frame(load_pickle(path) for path in sorted(Path(data_dir).rglob("results.pkl")))


def load_hosted_per_split(subset: str | None = "lite") -> pd.DataFrame:
    """The hosted TabArena-v0.1 per-split results (``results_per_split.csv`` of ``compare(plot=False)``)."""
    import tempfile

    from tabarena.contexts import TabArenaContext

    with tempfile.TemporaryDirectory() as tmp:
        TabArenaContext().compare(output_dir=tmp, subset=subset, plot=False)
        return pd.read_csv(Path(tmp) / "results_per_split.csv")


# --------------------------------------------------------------------------------------------------
# Entry points
# --------------------------------------------------------------------------------------------------
def audit_system(
    system_cls: type,
    config: dict | None = None,
    *,
    datasets: Iterable[str | ProbeData] = DEFAULT_DATASETS,
    checks: Iterable[str] = CHECKS,
    time_limit: float = 600,
    offline: bool = False,
    seed: int = 0,
    max_train_rows: int | None = 2000,
    max_test_rows: int | None = 1000,
    log: Callable[[str], None] = print,
) -> list[CheckResult]:
    """Run the audit and return every check result (see the module docstring)."""
    checks = tuple(checks)
    unknown = set(checks) - set(CHECKS)
    if unknown:
        raise ValueError(f"Unknown check(s) {sorted(unknown)}; valid: {CHECKS}")
    sut = SystemUnderAudit(system_cls=system_cls, config=dict(config or {}), time_limit=time_limit)
    results: list[CheckResult] = []
    if "source" in checks:
        results.append(scan_source(system_cls))
        log(format_result(results[-1]))
    probes = [c for c in checks if c not in ("source", "time")]
    for entry in datasets:
        data = (
            entry
            if isinstance(entry, ProbeData)
            else load_probe_data(entry, seed=seed, max_train_rows=max_train_rows, max_test_rows=max_test_rows)
        )
        if offline:
            if "egress" in checks:
                results.append(offline_egress(sut, data))
                log(format_result(results[-1]))
            continue
        if not probes:
            continue
        log(f"--- probing {data.name} ({data.problem_type}, {len(data.X_train)} train / {len(data.X_test)} test rows)")
        try:
            new = probe_dataset(sut, data, probes, seed=seed)
        except Exception as exc:
            new = [_probe_failure(sut, data.name, exc)]
        for result in new:
            log(format_result(result))
        results.extend(new)
    if "time" in checks and not offline:
        results.append(check_time(sut))
        log(format_result(results[-1]))
    return results


def format_result(result: CheckResult) -> str:
    """One aligned line per check result: verdict, check, dataset, summary."""
    return f"[{result.verdict:<4}] {result.check:<12} {result.dataset:<34} {result.summary}"


def _write_json(results: list[CheckResult], path: str, meta: dict) -> None:
    payload = {"meta": meta, "results": [asdict(r) for r in results]}
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, default=str))
    print(f"Wrote {path}")


def main(argv: list[str] | None = None) -> int:
    from tabarena.models._registry import assert_autogluon_resolves

    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--system", help="Registered system (method or display name).")
    parser.add_argument("--config", default=None, help="JSON init kwargs; default: the generator's first config.")
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--checks", nargs="+", default=list(CHECKS), choices=CHECKS)
    parser.add_argument("--time-limit", type=float, default=600, help="Seconds handed to every fit.")
    parser.add_argument("--offline", action="store_true", help="Block the network: source + egress only.")
    parser.add_argument("--seed", type=int, default=None, help="Default: a fresh seed, printed for reruns.")
    parser.add_argument("--max-train-rows", type=int, default=2000)
    parser.add_argument("--max-test-rows", type=int, default=1000)
    parser.add_argument(
        "--submitted-results",
        nargs="+",
        help="CSV(s) with dataset, fold, metric_error to compare with hosted (one per config).",
    )
    parser.add_argument("--json", help="Write every result (with details) to this path.")
    args = parser.parse_args(argv)
    if not args.system and not args.submitted_results:
        parser.error("pass --system and/or --submitted-results")
        return 2

    assert_autogluon_resolves()
    seed = args.seed if args.seed is not None else int(time.time()) % 2**31
    meta = {"system": args.system, "seed": seed, "checks": args.checks, "offline": args.offline}
    print(f"audit_system: seed {seed}")
    results: list[CheckResult] = []
    if args.submitted_results:
        hosted = load_hosted_per_split()
        for path in args.submitted_results:
            for result in compare_submitted_results(pd.read_csv(path), hosted, label=path):
                print(format_result(result))
                results.append(result)
    if args.system:
        system_cls, config = resolve_system(args.system)
        if args.config:
            config = json.loads(args.config)
        meta.update({"system_cls": f"{system_cls.__module__}.{system_cls.__qualname__}", "config": config})
        print(f"audit_system: {meta['system_cls']} config={config} time_limit={args.time_limit:g}s")
        results += audit_system(
            system_cls,
            config,
            datasets=args.datasets,
            checks=args.checks,
            time_limit=args.time_limit,
            offline=args.offline,
            seed=seed,
            max_train_rows=args.max_train_rows,
            max_test_rows=args.max_test_rows,
        )
    counts = {v: sum(r.verdict == v for r in results) for v in (FAIL, WARN, PASS, INFO, SKIP)}
    print("audit_system summary: " + ", ".join(f"{n} {v}" for v, n in counts.items() if n))
    if args.json:
        _write_json(results, args.json, meta)
    return 1 if counts[FAIL] else 0


if __name__ == "__main__":
    sys.exit(main())
