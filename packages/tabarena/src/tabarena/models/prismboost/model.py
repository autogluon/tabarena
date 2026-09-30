from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, ClassVar

import numpy as np
from autogluon.core.models import AbstractModel

from tabarena.models.prismboost._internal.preprocessing import PrismBoostPreprocessor

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.utils.config_utils import ConfigGenerator

logger = logging.getLogger(__name__)

_CLASSIFIER_ONLY_PARAMS = ("class_weight", "scale_pos_weight")
_REGRESSOR_UNSUPPORTED_PARAMS = (*_CLASSIFIER_ONLY_PARAMS, "second_order")
#: Wrapper-owned parameters, consumed here and never forwarded to the estimator.
_WRAPPER_PARAMS = (
    "numeric_scaler",
    "categorical_encoder",
    "ohe_max_cardinality",
    "max_n_estimators",
)
#: Boosting rounds tried when a validation set is available; the last rung is replaced by
#: ``max_n_estimators``. Doubling keeps the ladder's total cost under 2x its final fit.
_ITERATION_LADDER = (50, 100, 200, 400, 800)
#: Consecutive rungs allowed to not improve validation error before the ladder stops.
_ITERATION_PATIENCE = 1
#: Fraction of ``time_limit`` the ladder may spend, leaving room for predict and cleanup.
_TIME_BUDGET_FRACTION = 0.95


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

    **Boosting rounds come from the validation split.** PrismBoost has no eval-set or staged
    prediction API, so this wrapper does the next best thing within the ``_fit`` contract: with
    ``n_estimators`` left at ``"auto"`` it fits a doubling ladder of round counts, scores each on
    the ``X_val`` / ``y_val`` TabArena provides, and keeps the best. It is early stopping paid for
    by refitting -- the ladder costs under 2x its final rung -- and the chosen count goes into
    ``params_trained``, so a refit replays that count as an explicit value and fits once. An
    explicit ``n_estimators``, or a fit with no validation set, also goes straight to one fit.
    A ``staged_decision_function`` or an ``eval_set`` argument upstream would make this one fit.

    ``reg_lambda`` needs prismboost>=0.3.0. Without it an unregularized Newton leaf on a nearly
    pure node can drive the boosted scores far enough to saturate the softmax, which log loss
    scores as an infinite penalty on a single wrong row.

    ``num_cpus`` is accepted and unused: PrismBoost's C++ core is single-threaded (it links no
    OpenMP) and its Python backend is NumPy-level, so there is no thread argument to wire the
    budget to. A thread count upstream would let the wrapper honour it.
    """

    ag_key = "PRISMBOOST"
    ag_name = "PrismBoost"
    ag_priority = 65
    seed_name = "random_state"
    warmup_modules: ClassVar[tuple[str, ...]] = ("prismboost",)
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 8, "max_depth": 2, "max_n_estimators": 8}
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
        max_n_estimators = params["max_n_estimators"]
        for name in (*_WRAPPER_PARAMS, *unsupported):
            params.pop(name, None)
        # The C++ core implements the Newton criterion only, so first-order configs fall back to
        # the Python backend and everything else stays on the fast path.
        params["use_cpp"] = params.get("second_order") is not False

        X = self.preprocess(X, y=y, is_train=True)
        fit_kwargs = {} if sample_weight is None else {"sample_weight": sample_weight}

        # An explicit round count is honoured as given: that is how a refit reuses the count the
        # ladder chose (AutoGluon replays it through `params_trained`), and how a caller pins one.
        if X_val is None or y_val is None or params.get("n_estimators") != "auto":
            self.model = model_cls(**params).fit(X, y, **fit_kwargs)
            return

        X_val = self.preprocess(X_val)
        self._fit_iteration_ladder(
            model_cls=model_cls,
            params=params,
            max_n_estimators=max_n_estimators,
            X=X,
            y=y,
            X_val=X_val,
            y_val=y_val,
            fit_kwargs=fit_kwargs,
            start_time=start_time,
            time_limit=time_limit,
        )

    def _fit_iteration_ladder(
        self,
        *,
        model_cls,
        params: dict,
        max_n_estimators: int,
        X: np.ndarray,
        y: pd.Series,
        X_val: np.ndarray,
        y_val: pd.Series,
        fit_kwargs: dict,
        start_time: float,
        time_limit: float | None,
    ) -> None:
        """Fit increasing round counts and keep the one with the best validation error."""
        params = dict(params)
        params.pop("n_estimators", None)
        rungs = [n for n in _ITERATION_LADDER if n < max_n_estimators]
        rungs.append(max_n_estimators)

        best_error = float("inf")
        best_rounds = rungs[0]
        misses = 0
        last_rounds: int | None = None
        last_seconds = 0.0

        for rounds in rungs:
            if time_limit is not None and last_rounds is not None:
                # Fit time is close to linear in the round count; extrapolate from the last rung.
                projected = last_seconds * rounds / last_rounds
                elapsed = time.time() - start_time
                if elapsed + projected > _TIME_BUDGET_FRACTION * time_limit:
                    logger.log(
                        15,
                        f"\tPrismBoost: stopping the round ladder at {best_rounds} "
                        f"({rounds} rounds would not fit the remaining time budget).",
                    )
                    break
            rung_start = time.time()
            model = model_cls(n_estimators=rounds, **params).fit(X, y, **fit_kwargs)
            last_seconds = max(time.time() - rung_start, 1e-6)
            last_rounds = rounds

            error = self._validation_error(model, X_val, y_val)
            # `self.model is None` keeps the first rung even when its score is not finite, so a
            # degenerate validation score cannot leave the fit without a model.
            if self.model is None or error < best_error:
                best_error, best_rounds, self.model = error, rounds, model
                misses = 0
            else:
                misses += 1
                if misses > _ITERATION_PATIENCE:
                    break

        if self.model is None:  # every rung was cut by the time limit
            self.model = model_cls(n_estimators=rungs[0], **params).fit(X, y, **fit_kwargs)
            best_rounds = rungs[0]
        self.params_trained["n_estimators"] = best_rounds

    def _validation_error(self, model, X_val: np.ndarray, y_val: pd.Series) -> float:
        if self.problem_type == "regression":
            y_pred = model.predict(X_val)
        else:
            y_pred = self._align_proba(model, model.predict_proba(X_val))
        y_pred = self._convert_proba_to_unified_form(y_pred)
        return self.score_with_y_pred_proba(y=y_val, y_pred_proba=y_pred, as_error=True)

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
            # Capacity knobs stay at PrismBoost's data-adaptive "auto" rules. "auto" here also
            # means "let the validation ladder choose", and any explicit value pins the fit.
            "n_estimators": "auto",
            "max_n_estimators": 1600,
            # Chosen by varying only this knob through the bagged pipeline: mean rank over six
            # TabArena-Lite datasets is standard 2.00, robust 2.50, minmax 2.67,
            # quantile-normal 3.00, quantile-uniform 4.83, and the 25-config search agrees
            # (standard first of five, quantile-normal last). An earlier single-holdout sweep
            # picked quantile-normal; bagging reorders them, so the bagged result is the one
            # that counts. qsar-biodeg is the clearest case: 0.0871 -> 0.0773.
            "numeric_scaler": "standard",
            "categorical_encoder": "target",
            # L2 on leaf weights. PrismBoost defaults this to 0.0 to keep its published results
            # reproducible; 0.0 is the worst of the five settings measured over TabArena-Lite
            # (mean rank 3.67, worst case 58% off the best) and 10.0 the best on every problem
            # type (2.43, worst case 9%), so the benchmark default is set here rather than there.
            "reg_lambda": 10.0,
            "ohe_max_cardinality": 32,
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
        buffers and the growing ensemble, and the ladder holds two fitted models at once.
        Over-estimate rather than OOM.
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
