"""ChimeraBoost TabArena model: a pure-Python, CatBoost-inspired, numba-backed
oblivious gradient-boosting library (https://github.com/bbstats/chimeraboost).

Drop this package at `tabarena/.../models/chimeraboost/` in a fork of
autogluon/tabarena (see ../REGISTER.md). The model class MUST live in its own
file (not the run script) because TabArena pickles it.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

from autogluon.common.utils.resource_utils import ResourceManager
from autogluon.core.models import AbstractModel

# Import at module top rather than inside _fit: chimeraboost's import cost
# (numpy + numba module load, ~1.4s) would otherwise be charged to every
# job's reported train time. Tolerate absence so the registry row survives
# without the optional dependency (pip_extra); _fit raises a clear error.
try:
    from chimeraboost import ChimeraBoostClassifier, ChimeraBoostRegressor

    _CHIMERABOOST_IMPORT_ERROR = None
except ImportError as _exc:  # optional dependency not installed
    ChimeraBoostClassifier = ChimeraBoostRegressor = None
    _CHIMERABOOST_IMPORT_ERROR = _exc

if TYPE_CHECKING:
    import pandas as pd


def prefetch_weights() -> None:
    """Pre-compile chimeraboost's numba kernels (its analog of weight fetching).

    chimeraboost JIT-compiles its kernels on first use (~5-15s, disk-cached per
    environment). Without this, a fresh worker pays that compile inside the
    timed ``fit``/``predict`` sections, dwarfing the actual model work on small
    tasks. ``warmup()`` runs a few tiny synthetic fits covering every
    default-path kernel; predictions are bit-identical. Requires
    chimeraboost >= 0.14.1. On workers that do not share the prefetch cache,
    set ``CHIMERABOOST_WARMUP=1`` in the worker environment instead (runs the
    same warmup at import, which is outside the benchmark timers).
    """
    import chimeraboost

    chimeraboost.warmup()


class ChimeraBoostModel(AbstractModel):
    """ChimeraBoost as an AutoGluon/TabArena model (scikit-learn style API)."""

    ag_key = "CHIMERA"
    ag_name = "ChimeraBoost"
    seed_name = "random_state"  # AutoGluon injects the framework seed here

    def _preprocess(self, X: pd.DataFrame, is_train=False, **kwargs) -> pd.DataFrame:
        """Pass the frame straight to ChimeraBoost with categoricals marked by
        name. ChimeraBoost factorizes them with its native ordered-target-
        statistics encoding and routes NaN to a dedicated missing bin, so we do
        NOT label-encode or impute here (both would discard signal).
        """
        X = super()._preprocess(X, **kwargs)
        if is_train:
            # category-dtype columns (AutoGluon marks them; valid_raw_types keeps
            # them as 'category'); recorded once and reused at predict.
            self._cat_col_names = list(X.select_dtypes(include="category").columns)
        return X

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame = None,
        y_val: pd.Series = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: float = 0,
        verbosity: int = 2,
        **kwargs,
    ):
        start_time = time.time()
        if _CHIMERABOOST_IMPORT_ERROR is not None:
            raise ImportError(
                "ChimeraBoost requires the optional dependency 'chimeraboost' "
                "(pip install 'chimeraboost>=0.14.1')."
            ) from _CHIMERABOOST_IMPORT_ERROR
        if self.problem_type in ["regression"]:
            model_cls = ChimeraBoostRegressor
        else:  # 'binary' and 'multiclass'
            model_cls = ChimeraBoostClassifier

        X = self.preprocess(X, is_train=True)
        params = self._get_model_params()
        # Run on the CPU budget TabArena allocates (thread_count<0 => all cores).
        params["thread_count"] = num_cpus
        # Tuned configs may sample linear_leaves=True, which raises on multiclass;
        # None = auto (binary on, multiclass off) keeps the config valid everywhere.
        if self.problem_type == "multiclass" and params.get("linear_leaves") is True:
            params["linear_leaves"] = None
        self.model = model_cls(**params)

        cat = self._cat_col_names or None
        # Use TabArena's validation split for early stopping when provided (don't
        # carve a second holdout out of the training data); else ChimeraBoost
        # auto-splits internally via early_stopping=True.
        eval_set = None
        if X_val is not None and y_val is not None:
            X_val = self.preprocess(X_val)
            eval_set = (X_val, y_val)

        fit_kwargs = {}
        # Stop boosting once TabArena's fit budget runs out, leaving 5% headroom
        # for the final model build and scoring.
        if time_limit is not None:
            deadline = start_time + 0.95 * time_limit

            def _time_stop(iteration, train_loss, val_loss, model):
                return time.time() >= deadline

            fit_kwargs["callbacks"] = _time_stop

        self.model.fit(X, y, cat_features=cat, eval_set=eval_set, **fit_kwargs)

    def _set_default_params(self):
        default_params = {
            # Cap only: early stopping picks the real count and the auto learning
            # rate is pinned at 0.1 under ES, so a high cap is LR-neutral headroom.
            "n_estimators": 10000,
            "early_stopping": True,
        }
        for param, val in default_params.items():
            self._set_default_param_value(param, val)

    def _get_default_auxiliary_params(self) -> dict:
        default_auxiliary_params = super()._get_default_auxiliary_params()
        default_auxiliary_params.update({"valid_raw_types": ["int", "float", "category"]})
        return default_auxiliary_params

    @classmethod
    def supported_problem_types(cls) -> list[str] | None:
        return ["binary", "multiclass", "regression"]

    def _get_default_resources(self) -> tuple[int, int]:
        # Physical cores only (matches RealMLP/XRFM); ChimeraBoost is CPU-only.
        num_cpus = ResourceManager.get_cpu_count(only_physical_cores=True)
        return num_cpus, 0

    def _estimate_memory_usage(self, X: pd.DataFrame, **kwargs) -> int:
        return self.estimate_memory_usage_static(
            X=X,
            problem_type=self.problem_type,
            num_classes=self.num_classes,
            hyperparameters=self._get_model_params(),
            **kwargs,
        )

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = 1,
        **kwargs,
    ) -> int:
        """Conservative peak-fit RAM estimate (bytes) for fold-parallel scheduling.

        ChimeraBoost is a CPU GBDT; peak memory is dominated by O(n_samples *
        n_features) terms — the input matrix, the quantized bin codes, and a few
        per-row stat buffers (gradients, hessians, predictions, the validation
        copy). The oblivious trees themselves are negligible (2**depth leaves *
        n_estimators * a few bytes). We deliberately over-estimate (a 3x factor
        on the input matrix + a 1 GB baseline) so the scheduler packs fewer folds
        rather than risk OOM.
        """
        n, p = int(X.shape[0]), int(X.shape[1])
        k = max(int(num_classes or 1), 1)
        cell = 8  # float64 / object-pointer width
        data = n * p * cell  # input matrix (object array when cats present)
        binned = n * p * 2  # quantized bin codes (uint8/uint16)
        stats = n * k * cell * 6  # grad / hess / pred / weight / val buffers
        hist = p * 256 * 2 * cell  # transient per-level histograms
        baseline = 1_000_000_000  # python + numba + autogluon overhead
        return int(baseline + 3 * data + binned + stats + hist)

    @classmethod
    def _class_tags(cls) -> dict:
        return {"can_estimate_memory_usage_static": True}
