from __future__ import annotations

import base64
import hashlib
import io
import os
import time
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from tabarena.benchmark.exec_models import ExternalSystemModel

if TYPE_CHECKING:
    from autogluon.core.metrics import Scorer

    from tabarena.benchmark.task.metadata import ValidationMetadata

# Endpoint and key come from the environment so a run never carries a credential in its config.
_URL_ENV = "CHAKRA_TAB_URL"
_KEY_ENV = "CHAKRA_TAB_KEY"
_SCHEME_ENV = (
    "CHAKRA_TAB_AUTH_SCHEME"  # "Bearer" for the public API (default); the provider's raw endpoints use "Api-Key"
)
_DEFAULT_URL = "https://api.yhatlabs.com/v1/tabular/predict"
_METRICS = {"log_loss": "log_loss", "roc_auc": "roc_auc", "rmse": "rmse", "root_mean_squared_error": "rmse"}


def _parquet_b64(df: pd.DataFrame) -> dict:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return {"format": "parquet_b64", "bytes": base64.b64encode(buf.getvalue()).decode()}


class ChakraTabSystemModel(ExternalSystemModel):
    """Chakra-Tab, YHat Labs' hosted tabular prediction API, benchmarked as a system.

    The API fits a table and predicts rows in one call (``fit_predict``): validation, bagging and
    ensembling happen behind it, so TabArena hands it the raw frames and records what comes back.
    The fit stores the training table; the first ``predict`` / ``predict_proba`` on a set of rows
    makes the call, and both share its result.

    Init hyperparameters (per-config knobs for the system generator):

    * ``preset`` -- ``"medium"`` (default: 8-fold bagging, every component in-context) or
      ``"full"`` (8-fold bagging with component fine-tuning).

    ``time_limit`` is forwarded to the API as the fit budget. The endpoint is read from
    ``CHAKRA_TAB_URL`` (default: the public API) and the key from ``CHAKRA_TAB_KEY``.

    API documentation: https://yhatlabs.com
    """

    uses_ray = False

    def __init__(self, *, preset: str = "medium", url: str | None = None, **kwargs):
        super().__init__(**kwargs)
        self.preset = preset
        self.url = url or os.environ.get(_URL_ENV, _DEFAULT_URL)
        self._train: pd.DataFrame | None = None
        self._target: str | None = None
        self._problem_type: str | None = None
        self._metric: str | None = None
        self._time_limit: float | None = None
        self._cache: dict[str, dict] = {}
        self.fit_info: dict | None = None

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
        X[target_name] = y.to_numpy() if hasattr(y, "to_numpy") else y
        self._train = X
        self._target = target_name
        self._problem_type = problem_type
        self._metric = _METRICS.get(getattr(eval_metric, "name", str(eval_metric)))
        self._time_limit = time_limit
        return self

    def _call(self, X_test: pd.DataFrame) -> dict:
        import requests

        key = hashlib.sha256(pd.util.hash_pandas_object(X_test, index=False).to_numpy().tobytes()).hexdigest()
        if key in self._cache:
            return self._cache[key]
        body = {
            "op": "fit_predict",
            "train": _parquet_b64(self._train),
            "target": self._target,
            "test": _parquet_b64(X_test.reset_index(drop=True)),
            "preset": self.preset,
            "problem_type": self._problem_type,
            "eval_metric": self._metric,
            "time_limit": self._time_limit,
        }
        headers = {"Authorization": f"{os.environ.get(_SCHEME_ENV, 'Bearer')} {os.environ[_KEY_ENV]}"}
        out = None
        for attempt in range(4):
            r = requests.post(self.url, json=body, headers=headers, timeout=7200)
            if r.status_code == 200 and "error" not in r.json():
                out = r.json()
                break
            time.sleep(30 * (attempt + 1))
        if out is None:
            raise RuntimeError(f"Chakra-Tab API failed: {r.status_code} {r.text[:300]}")
        self.fit_info = out.get("fit")
        self._cache = {key: out}
        return out

    def _predict(self, X: pd.DataFrame) -> pd.Series:
        out = self._call(X)
        if self._problem_type == "regression":
            return pd.Series(np.asarray(out["predictions"], dtype=float), index=X.index)
        return pd.Series(out["predictions"], index=X.index)

    def _predict_proba(self, X: pd.DataFrame) -> pd.DataFrame:
        out = self._call(X)
        return pd.DataFrame(np.asarray(out["probabilities"], dtype=float), index=X.index, columns=out["classes"])

    def cleanup(self):
        self._train = None
        self._cache = {}
