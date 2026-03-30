from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Lasso
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class LassoFeatureSelector(AbstractFeatureSelector):
    """Lasso Feature Selection.

    Reference: Tibshirani, Robert. "Regression shrinkage and selection via the lasso." Journal of the Royal Statistical Society Series B: Statistical Methodology 58.1 (1996): 267-288.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """

    name = "LassoFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        numeric_imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)

        lasso = Lasso(random_state=1)
        lasso.fit(X_imputed, y)
        scores = lasso.coef_
        return dict(zip(X.columns, scores))
