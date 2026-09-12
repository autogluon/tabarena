from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.core.models import AbstractModel

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd


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
    """

    ag_key = "TA-APLR"
    ag_name = "aplr"
    ag_priority = 65
    seed_name = "random_state"
    _supported_problem_types = ["binary", "multiclass", "regression"]
    default_resources_physical_cores_only = True

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
        del time_limit, num_gpus, kwargs

        import numpy as np
        import pandas as pd
        from aplr import APLRClassifier, APLRRegressor

        params = self._get_model_params()
        params["n_jobs"] = num_cpus
        model_cls = APLRRegressor if self.problem_type == "regression" else APLRClassifier

        self.model = model_cls(**params)

        if X_val is not None and y_val is not None:
            X_full = pd.concat([X, X_val], ignore_index=True)
            y_full = pd.concat([y, y_val], ignore_index=True)
            X_full = self.preprocess(X_full, y=y_full)
            cv_observations = np.column_stack(
                [
                    np.concatenate(
                        [
                            np.ones(len(X), dtype=int),
                            -np.ones(len(X_val), dtype=int),
                        ]
                    )
                ]
            )
            self.model.fit(X_full, y_full, cv_observations=cv_observations)
            return

        X = self.preprocess(X, y=y)
        self.model.fit(X, y)

    def _set_default_params(self):
        pass  # Intentionally keeps APLR default parameters

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        X = self.preprocess(X)
        predictions = self.model.predict(X) if self.problem_type == "regression" else self.model.predict_proba(X)
        return self._convert_proba_to_unified_form(predictions)

    def _predict(self, X: pd.DataFrame, **kwargs):
        return self.model.predict(self.preprocess(X))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}
