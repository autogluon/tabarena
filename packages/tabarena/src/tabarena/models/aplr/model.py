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
        del X_val, y_val, time_limit, num_gpus, kwargs

        from aplr import APLRClassifier, APLRRegressor

        params = self._get_model_params()
        params["n_jobs"] = num_cpus
        model_cls = APLRRegressor if self.problem_type == "regression" else APLRClassifier

        self.model = model_cls(**params)
        X = self.preprocess(X, y=y)
        self.model.fit(X, y)

    def _set_default_params(self):
        for param, value in {"max_interaction_level": 1}.items():
            self._set_default_param_value(param, value)

    def _predict_proba(self, X: pd.DataFrame, **kwargs) -> np.ndarray:
        return self._convert_proba_to_unified_form(self.model.predict_proba(self.preprocess(X)))

    def _predict(self, X: pd.DataFrame, **kwargs):
        return self.model.predict(self.preprocess(X))

    def _more_tags(self) -> dict:
        return {"can_refit_full": True}


class APLRTwoWayIntModel(APLRModel):
    """APLR configured to allow two-way interaction terms."""

    ag_key = "TA-APLR_TWO_WAY_INT"
    ag_name = "aplr_two_way_int"


class APLRDeepIntModel(APLRModel):
    """APLR configured to allow deeper interaction terms."""

    ag_key = "TA-APLR_DEEP_INT"
    ag_name = "aplr_deep_int"

    def _set_default_params(self):
        self._set_default_param_value("max_interaction_level", 3)
