from __future__ import annotations

import os
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    from collections.abc import Callable

    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

# The key is the client's own variable, read at call time, so a run never carries a credential in its config.
_KEY_ENV = "TABPFN_TOKEN"
_MAX_ATTEMPTS = 3


def _model_limit():
    """The API's per-call caps (the client's settings call; ``None`` when the server sends none)."""
    from tabpfn_client.client import ServiceClient

    settings = ServiceClient.get_settings()
    return None if settings is None else settings.max_model_limit


def _timeout_cap(effort: str) -> float | None:
    """The longest ``thinking_timeout_s`` the API takes at ``effort`` (``None`` when the server does not say).

    The maximum the server's settings give for ``effort`` when they give one, else the longest any fit
    takes, as the client checks it before the upload.
    """
    from tabpfn_client.client import ServiceClient

    settings = ServiceClient.get_settings()
    if settings is None:
        return None
    limits = (settings.thinking_model_limits or {}).get(settings.thinking_default_model_version)
    cap = None if limits is None or limits.max_timeout_secs is None else limits.max_timeout_secs.get(effort)
    return settings.thinking_max_timeout_secs if cap is None else cap


class TabPFN35ThinkingSystemModel(ExternalSystemModel):
    """TabPFN-3.5-Thinking, Prior Labs' hosted tabular prediction system, benchmarked as a system.

    TabPFN-3.5-Thinking is served through Prior Labs' API and called with the official client,
    ``tabpfn-client``, in thinking mode: ``TabPFNClassifier`` / ``TabPFNRegressor`` with
    ``thinking_effort``. The fit uploads the training features and the target and waits for the
    server's fit; ``predict`` / ``predict_proba`` upload the test features alone and return the
    server's predictions. Fit and prediction are separate calls, so ``time_train_s`` holds the fit.

    Init hyperparameters (per-config knobs for the system generator):

    * ``effort`` -- the thinking effort, ``"xhigh"`` by default. The API's levels are ``"micro"``, ``"mini"``,
      ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` and ``"max"``.

    The task's metric is forwarded as ``thinking_metric`` by its scorer name, which the API accepts as is,
    and the split's ``random_state`` as ``random_state``. A task's group and time columns, when it has them
    (``validation_metadata.group_on`` / ``time_on`` / ``group_time_on``, as on BeyondArena's grouped and
    temporal tasks), are named to the client as ``group_col`` / ``time_col`` / ``group_time_col``; the
    columns themselves are part of ``X``. ``time_limit`` is forwarded as ``thinking_timeout_s``,
    capped at the longest the API takes at the chosen effort (the server reports it in its settings,
    and the client refuses a longer value). The hardware is the provider's, so ``num_cpus`` /
    ``num_gpus`` / ``memory_limit`` are not forwarded.

    Classification labels held as Python objects of one non-string type (integers in an ``object``
    column) are converted to that type before the fit, as the client accepts no other ``object``
    labels than strings. A prediction larger than one API call allows (the server's caps on test rows,
    on test rows times training rows and on test cells) is sent as several calls over consecutive row
    blocks against the same fitted model.

    The key is read from ``TABPFN_TOKEN`` at fit time and handed to the client with
    ``set_access_token``, so a token cached by an earlier ``interactive_login`` is never used. The
    endpoint is the client's default (the public API); ``TABPFN_CLIENT_API_URL`` overrides it. The
    client retries its own requests and waits for a fit at most its poll timeout (by default the
    server's, which covers the longest fit the server takes). On top, a call that ends in a transport
    error or a 5xx / 429 the client gave up on is repeated, up to ``_MAX_ATTEMPTS`` calls in all, with
    a growing pause; any other error fails at once. ``get_metadata`` returns the client version, the
    server's fit and prediction timings, what the server reported with the last prediction (the
    service version) and each call's attempts and wall time, which the runner stores with the result.

    API documentation: https://docs.priorlabs.ai/capabilities/thinking-mode
    """

    warmup_modules: ClassVar[tuple[str, ...]] = ("tabpfn_client",)

    def __init__(self, *, effort: str = "xhigh", **kwargs):
        super().__init__(**kwargs)
        self.effort = effort
        self._estimator = None
        self._metric: str | None = None
        self._split_columns: dict[str, str | list[str]] = {}
        self._timeout_s: float | None = None
        self._n_train = 0
        self._api_calls: list[dict] = []

    def _fit_system(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        *,
        target_name: str,
        problem_type: str,
        eval_metric: Scorer,
        validation_metadata: ValidationMetadata,
        num_cpus: int | None,
        num_gpus: int | None,
        memory_limit: float | None,
        time_limit: float | None,
        random_state: int | None,
    ):
        key = os.environ.get(_KEY_ENV)
        if not key:
            raise RuntimeError(f"TabPFN-3.5-Thinking needs its API key in the {_KEY_ENV} environment variable.")
        from tabpfn_client import TabPFNClassifier, TabPFNRegressor, set_access_token

        set_access_token(key)
        self._metric = eval_metric.name
        estimator_cls = TabPFNRegressor if problem_type == "regression" else TabPFNClassifier
        if problem_type != "regression" and y.dtype == object:
            y = y.infer_objects()
        split_columns = {
            "group_col": validation_metadata.group_on,
            "time_col": validation_metadata.time_on,
            "group_time_col": validation_metadata.group_time_on,
        }
        self._split_columns = {k: v for k, v in split_columns.items() if v is not None}
        self._timeout_s = None if time_limit is None else float(time_limit)
        cap = None if self._timeout_s is None else _timeout_cap(self.effort)
        if cap is not None:
            self._timeout_s = min(self._timeout_s, cap)
        self._estimator = estimator_cls(
            thinking_effort=self.effort,
            thinking_timeout_s=self._timeout_s,
            thinking_metric=self._metric,
            random_state=0 if random_state is None else int(random_state),
            **self._split_columns,
        )
        self._n_train = len(X)
        self._call("fit", len(X), lambda: self._estimator.fit(X, y))
        return self

    def _row_blocks(self, X: pd.DataFrame) -> list[pd.DataFrame]:
        """``X`` in consecutive row blocks that each fit one prediction call (one block when it fits already)."""
        limit = _model_limit()
        if limit is None:
            return [X]
        rows = min(
            limit.test_set_max_rows,
            limit.predict_row_pairs_budget // max(self._n_train, 1),
            limit.test_set_max_cells // max(X.shape[1], 1),
        )
        rows = max(rows, 1)
        return [X.iloc[i : i + rows] for i in range(0, len(X), rows)]

    def _predict_blocks(self, what: str, X: pd.DataFrame, predict: Callable[[pd.DataFrame], object]) -> np.ndarray:
        blocks = self._row_blocks(X)
        out = [np.asarray(self._call(what, len(block), lambda b=block: predict(b))) for block in blocks]
        return out[0] if len(out) == 1 else np.concatenate(out, axis=0)

    def _call(self, what: str, rows: int, call: Callable[[], object]) -> object:
        """Run one client call, repeating it after a transient failure (see the class docstring)."""
        import httpx
        from tabpfn_client.errors import CappedRetryableServerError, EmptyResponseError, RetryableServerError

        transient = (RetryableServerError, CappedRetryableServerError, EmptyResponseError, httpx.TransportError)
        failure, start, attempt = None, time.monotonic(), 1
        while True:
            try:
                out = call()
            except transient as exc:
                failure = f"{type(exc).__name__}: {str(exc)[:300]}"
                if attempt < _MAX_ATTEMPTS:
                    time.sleep(30 * attempt)
                    attempt += 1
                    continue
                self._record(what, rows, attempt, start, failure)
                raise RuntimeError(
                    f"TabPFN-3.5-Thinking API {what} failed after {attempt} attempts: {failure}"
                ) from exc
            self._record(what, rows, attempt, start, failure)
            return out

    def _record(self, what: str, rows: int, attempts: int, start: float, failure: str | None) -> None:
        self._api_calls.append(
            {"call": what, "rows": rows, "attempts": attempts, "wall_s": time.monotonic() - start, "failure": failure},
        )

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        pred = self._predict_blocks("predict", X, self._estimator.predict)
        return pd.Series(pred, index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        proba = self._predict_blocks("predict_proba", X, self._estimator.predict_proba)
        return pd.DataFrame(np.asarray(proba, dtype=float), index=X.index, columns=list(self._estimator.classes_))

    def get_metadata(self) -> dict:
        """What the API reported about the fit and the last prediction, and each call, for ``results.pkl``.

        ``api_fit_s`` / ``api_predict_s`` lift the server-side fit and prediction times out of the
        client's ``get_timings()`` (the last prediction call's, when a prediction took several);
        ``api_server_info`` is what the server returned with the last prediction, which pins the version
        of the service that produced a result; ``api_split_columns`` are the group and time columns named
        to the client (empty when the task has none); ``thinking_timeout_s`` is the time limit sent with the fit.
        """
        from tabpfn_client import __version__ as client_version

        est = self._estimator
        timings = est.get_timings() if est is not None else {}
        return {
            "client": f"tabpfn-client {client_version}",
            "effort": self.effort,
            "thinking_metric": self._metric,
            "thinking_timeout_s": self._timeout_s,
            "api_model_id": None if est is None else str(getattr(est, "model_id_", None)),
            "api_timings": timings,
            "api_fit_s": (timings.get("fit") or {}).get("fit_s"),
            "api_predict_s": (timings.get("predict") or {}).get("predict_s"),
            "api_server_info": None if est is None else est._last_meta,
            "api_split_columns": self._split_columns,
            "api_calls": self._api_calls,
        }

    def cleanup(self):
        self._estimator = None
