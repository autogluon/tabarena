from __future__ import annotations

import inspect
import logging
import time
from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel
from autogluon.core.utils.exceptions import TimeLimitExceeded

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

logger = logging.getLogger(__name__)


class APLRModel(AbstractModel):
    """Automatic Piecewise Linear Regression (APLR).

    Description: APLR builds predictive and interpretable regression or classification machine
    learning models in Python, using the Automatic Piecewise Linear Regression (APLR)
    methodology developed by Mathias von Ottenbreit. APLR often rivals tree-based
    methods in predictive accuracy, while offering smoother, more interpretable predictions.

    Paper: Automatic piecewise linear regression
    Authors: Mathias von Ottenbreit and Riccardo De Bin
    Codebase: https://github.com/ottenbreit-data-science/aplr
    License: MIT

    The fit time limit AutoGluon hands to ``_fit`` is forwarded to APLR's ``time_limit``
    constructor parameter, added upstream in https://github.com/ottenbreit-data-science/aplr/pull/18.
    APLR releases up to 10.26.0 lack it, ignore the budget and boost until early stopping; the
    wrapper logs a warning in that case.
    """

    ag_key = "TA-APLR"
    ag_name = "aplr"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True

    # Share of AutoGluon's remaining fit budget handed to APLR. The rest covers the wrapper's own
    # bookkeeping around the C++ fit and AutoGluon's validation predictions afterwards.
    _TIME_LIMIT_FRACTION = 0.95

    def _fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
        time_limit: float | None = None,
        num_cpus: int = 1,
        num_gpus: int = 0,
        **kwargs,
    ):
        del num_gpus, kwargs
        time_start = time.time()

        import numpy as np
        import pandas as pd
        from aplr import APLRClassifier, APLRRegressor

        params = self._get_model_params()
        params["n_jobs"] = num_cpus
        model_cls = APLRRegressor if self.problem_type == "regression" else APLRClassifier

        # Preprocess before the budget is computed, so APLR gets the time that is actually left.
        fit_kwargs: dict = {}
        if X_val is not None and y_val is not None:
            X_fit = pd.concat([X, X_val], ignore_index=True)
            y_fit = pd.concat([y, y_val], ignore_index=True)
            X_fit = self.preprocess(X_fit, y=y_fit)
            fit_kwargs["cv_observations"] = np.column_stack(
                [
                    np.concatenate(
                        [
                            np.ones(len(X), dtype=int),
                            -np.ones(len(X_val), dtype=int),
                        ]
                    )
                ]
            )
        else:
            X_fit = self.preprocess(X, y=y)
            y_fit = y

        if time_limit is not None:
            aplr_time_limit = self._aplr_time_limit(model_cls, time_limit=time_limit, time_start=time_start)
            if aplr_time_limit is not None:
                params["time_limit"] = aplr_time_limit

        self.model = model_cls(**params)
        self.model.fit(X_fit, y_fit, **fit_kwargs)

    @classmethod
    def _aplr_time_limit(cls, model_cls: type, *, time_limit: float, time_start: float) -> float | None:
        """Return the wall-clock budget in seconds to hand to APLR, or None if it cannot take one.

        APLR splits the budget across its inner validation folds and, for classification, across
        the one-vs-rest logit models, and stops boosting once a share is used up, so the fit ends
        within the budget. The time already spent since ``time_start`` (preprocessing) is deducted
        and a small fraction is kept back for the work around the C++ fit. Raises
        ``TimeLimitExceeded`` when nothing is left. Installed APLR versions without the
        ``time_limit`` parameter get a warning and run unbounded.
        """
        if "time_limit" not in inspect.signature(model_cls.__init__).parameters:
            logger.warning(
                "\tThe installed aplr package has no `time_limit` parameter, so this fit ignores the time limit. "
                "Install an APLR version with `time_limit` support to enforce it."
            )
            return None
        time_left = time_limit - (time.time() - time_start)
        if time_left <= 0:
            raise TimeLimitExceeded
        return time_left * cls._TIME_LIMIT_FRACTION

    def _set_default_params(self):
        pass  # Intentionally keeps APLR default parameters

    # Peak-memory model, calibrated on single child-fold fits of the TabArena-v0.1 tasks on
    # 2026-09-10. Binary and regression fits peaked at 6.2 to 6.9 dense float64 copies of the
    # expanded matrix (Amazon_employee_access, 19k rows x 8.1k one-hot columns: 7.9 GB;
    # kddcup09_appetency, 29k x 17.8k: 26.4 GB; every numeric-only task stayed below 1.4 GB).
    # Multiclass fits (SDSS17 with 3 and 6 classes) peaked at about 3.5 + 3.45 * n_classes copies,
    # independent of the thread count. The constants below round those up by roughly 10 percent.
    _MEMORY_COPIES_BINARY = 7.0
    _MEMORY_COPIES_MULTICLASS_BASE = 4.5
    _MEMORY_COPIES_PER_CLASS = 3.6
    _MEMORY_BASELINE_BYTES = 1024**3

    @classmethod
    def _estimate_memory_usage_static(
        cls,
        *,
        X: pd.DataFrame,
        hyperparameters: dict | None = None,
        num_classes: int | None = 1,
        **kwargs,
    ) -> int:
        """Estimate the peak fit RAM in bytes so AutoGluon caps the bagging folds it fits in parallel.

        APLR one-hot encodes every categorical column inside its C++ core into dense float64
        columns and appends a missing-value indicator for each numeric column with NaNs, so the
        footprint follows the width of that expanded matrix rather than the frame AutoGluon hands
        over: a small frame with a few high-cardinality categoricals can need tens of GB per fold.
        Multiclass targets fit one logit model per class and keep them all in memory, so the copy
        count grows linearly with the number of classes. Hyperparameters and the thread count do
        not move the peak (the deepest-interaction, smallest-split corner of the search space and
        8 threads measured equal or lower) and are ignored.
        """
        import pandas as pd

        n_rows = int(X.shape[0])
        expanded_width = 0
        for col in X.columns:
            s = X[col]
            if isinstance(s.dtype, pd.CategoricalDtype) or pd.api.types.is_string_dtype(s) or s.dtype == object:
                expanded_width += int(s.nunique(dropna=False))
            else:
                expanded_width += 1 + int(s.isna().any())
        n_classes = int(num_classes) if num_classes is not None else 1
        if n_classes > 2:
            copies = cls._MEMORY_COPIES_MULTICLASS_BASE + cls._MEMORY_COPIES_PER_CLASS * n_classes
        else:
            copies = cls._MEMORY_COPIES_BINARY
        dense_copy_bytes = n_rows * expanded_width * 8
        return int(copies * dense_copy_bytes + cls._MEMORY_BASELINE_BYTES)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X)
        predictions = self.model.predict(X) if self.problem_type == "regression" else self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(predictions)

    def _predict(self, X: pd.DataFrame, **kwargs):
        return self.model.predict(self.preprocess(X))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
