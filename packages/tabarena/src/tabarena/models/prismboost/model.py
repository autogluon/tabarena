from __future__ import annotations

import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.models import AbstractModel

from tabarena.models.prismboost._internal.preprocessing import PrismBoostPreprocessor

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.utils.config_utils import ConfigGenerator

_CLASSIFIER_ONLY_PARAMS = ("class_weight", "scale_pos_weight")
_REGRESSOR_UNSUPPORTED_PARAMS = (*_CLASSIFIER_ONLY_PARAMS, "second_order")
#: Share of the remaining budget handed to the library, leaving room for the validation scoring
#: inside the fit and the predict that follows it.
_TIME_BUDGET_FRACTION = 0.95
#: Floor on the budget handed over, so an already-exhausted limit still produces a usable model
#: rather than an empty one.
_MIN_FIT_SECONDS = 5.0
#: Wrapper-owned parameters, consumed here and never forwarded to the estimator.
_WRAPPER_PARAMS = ("numeric_scaler", "categorical_encoder", "ohe_max_cardinality")


class PrismBoostModel(AbstractModel):
    """PrismBoost: gradient boosting with SEFR oblique splits.

    Paper: PrismBoost (forthcoming)
    Authors: Hamidreza Keshavarz, Reza Rawassizadeh
    Codebase: https://github.com/PrismBoost/PrismBoost
    License: MIT

    PrismBoost's estimators take a dense finite float64 matrix, and a SEFR hyperplane is
    scale-sensitive, so this wrapper owns the encoding. ``numeric_scaler`` and
    ``categorical_encoder`` are hyperparameters rather than constants because PrismBoost's own
    PMLB study searched both per dataset; ``_internal/preprocessing.py`` says why neither has a
    safe fixed value.

    ``_fit`` is a single fit. The round count comes from early stopping on the ``X_val`` /
    ``y_val`` the harness provides, the way the other boosting wrappers here work:
    ``n_estimators`` is the cap and ``early_stopping_rounds`` the patience, both requiring
    prismboost>=0.4.0 for its ``eval_set`` support. Without a validation set (refit, holdout) the
    configured ``n_estimators`` is used as-is, and ``params_trained`` carries the count early
    stopping selected so a refit reproduces it.

    ``time_limit`` is passed through as prismboost's ``fit(time_limit=...)`` (>=0.5.0), minus the
    preprocessing already done and a 5% margin for scoring the validation split and predicting.
    The library checks it after each boosting stage and keeps the stages fitted so far, so the
    budget bounds the loop rather than guaranteeing a deadline: one stage on a very large table
    can still overshoot, which is the same contract the other boosters' callbacks offer.

    ``num_cpus`` is accepted and unused: PrismBoost's C++ core is single-threaded (it links no
    OpenMP) and its Python backend is NumPy-level, so there is no thread argument to wire the
    budget to. Recorded here as an upstream ask.
    """

    ag_key = "PRISMBOOST"
    ag_name = "PrismBoost"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("prismboost",)
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 8, "max_depth": 2}
    _supported_problem_types = ["binary", "multiclass", "regression"]
    _default_auxiliary_params_extra = {"valid_raw_types": ["int", "float", "category"]}
    default_resources_physical_cores_only = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._preprocessor: PrismBoostPreprocessor | None = None

    # --- preprocessing -------------------------------------------------------------------

    def _preprocess(self, X: pd.DataFrame, is_train: bool = False, y=None, **kwargs) -> np.ndarray:
        X = super()._preprocess(X, **kwargs)
        if is_train:
            params = self._get_model_params()
            self._preprocessor = PrismBoostPreprocessor(
                problem_type=self.problem_type,
                numeric_scaler=params["numeric_scaler"],
                categorical_encoder=params["categorical_encoder"],
                ohe_max_cardinality=params["ohe_max_cardinality"],
            )
            return self._preprocessor.fit_transform(X, y)
        if self._preprocessor is None:
            raise RuntimeError("PrismBoostModel preprocessor is not fitted.")
        return self._preprocessor.transform(X)

    # --- fit -----------------------------------------------------------------------------

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        sample_weight: np.ndarray | None = None,
        **kwargs,
    ) -> None:
        del num_cpus, num_gpus, kwargs  # single-threaded CPU library
        start_time = time.time()
        from prismboost import PrismBoostClassifier, PrismBoostRegressor

        if self.problem_type == "regression":
            model_cls, unsupported = PrismBoostRegressor, _REGRESSOR_UNSUPPORTED_PARAMS
        else:
            model_cls, unsupported = PrismBoostClassifier, ()

        params = self._get_model_params()
        for name in (*_WRAPPER_PARAMS, *unsupported):
            params.pop(name, None)
        # The C++ core implements the Newton criterion only, so first-order configs fall back to
        # the Python backend and everything else stays on the fast path.
        params["use_cpp"] = params.get("second_order") is not False

        X = self.preprocess(X, y=y, is_train=True)
        fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}

        if X_val is not None and y_val is not None:
            fit_kwargs["eval_set"] = (self.preprocess(X_val), np.asarray(y_val))
        else:
            # No validation set to stop on, so the configured cap is the round count.
            params.pop("early_stopping_rounds", None)

        if time_limit is not None:
            # Preprocessing is already spent; leave a margin for scoring the validation split
            # and for the predict that follows the fit.
            remaining = (time_limit - (time.time() - start_time)) * _TIME_BUDGET_FRACTION
            fit_kwargs["time_limit"] = max(remaining, _MIN_FIT_SECONDS)

        self.model = model_cls(**params).fit(X, y, **fit_kwargs)
        best = getattr(self.model, "best_iteration_", None)
        if best:
            self.params_trained["n_estimators"] = int(best)

    def _align_proba(self, model, proba: np.ndarray) -> np.ndarray:
        """Widen a child's probabilities to the task's full class set.

        A bagged fold whose training rows miss a rare class produces one column too few, which
        every downstream consumer reads as a different class order. Put the columns back where
        the task expects them and leave the missing class at probability zero.
        """
        num_classes = self.num_classes
        if num_classes is None or proba.ndim != 2 or proba.shape[1] == num_classes:
            return proba
        full = np.zeros((proba.shape[0], num_classes), dtype=np.float64)
        for column, label in enumerate(np.asarray(model.classes_, dtype=int)):
            full[:, label] = proba[:, column]
        return full

    def _predict_proba(self, X, **kwargs) -> np.ndarray:
        X = self.preprocess(X, **kwargs)
        if self.problem_type == "regression":
            return self.model.predict(X)
        return self._convert_proba_to_unified_form(
            self._align_proba(self.model, self.model.predict_proba(X)),
        )

    # --- configuration -------------------------------------------------------------------

    def _set_default_params(self) -> None:
        default_params = {
            # A high cap, as the other boosting wrappers here use: early stopping picks the real
            # count and `time_limit` bounds the work, so the cap is headroom rather than a budget.
            "n_estimators": 10000,
            "early_stopping_rounds": 50,
            # Chosen by varying only this knob through the bagged pipeline: mean rank over six
            # TabArena-Lite datasets is standard 2.00, robust 2.50, minmax 2.67,
            # quantile-normal 3.00, quantile-uniform 4.83.
            "numeric_scaler": "standard",
            "categorical_encoder": "target",
            "ohe_max_cardinality": 32,
            # L2 on leaf weights. PrismBoost defaults this to 0.0 to keep its published results
            # reproducible; 0.0 is the worst of five settings measured over TabArena-Lite and
            # 10.0 the best on every problem type, so the benchmark default is set here.
            "reg_lambda": 10.0,
        }
        for param, value in default_params.items():
            self._set_default_param_value(param, value)

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        num_classes: int | None = 1,
        hyperparameters: dict | None = None,
        **kwargs,
    ) -> int:
        """Conservative peak-fit RAM (bytes) so bagging folds are packed safely.

        The C++ core keeps a dense float64 copy of the encoded matrix plus gradient/hessian
        buffers and the growing ensemble. Over-estimate rather than OOM.
        """
        hyperparameters = hyperparameters or {}
        n_classes = max(int(num_classes or 1), 1)
        n_columns = PrismBoostPreprocessor.estimate_n_columns(
            X,
            ohe_max_cardinality=hyperparameters.get("ohe_max_cardinality", 32),
            num_classes=n_classes,
        )
        n_rows = int(X.shape[0])
        cell = 8
        data = n_rows * n_columns * cell
        stats = n_rows * n_classes * cell * 6
        baseline = 512 * 1024 * 1024
        return int(baseline + 4 * data + stats)

    @classmethod
    def config_generator(cls) -> ConfigGenerator:
        from tabarena.models.prismboost.hpo import gen_prismboost

        return gen_prismboost
