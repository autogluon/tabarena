import logging

import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import Lasso

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class LassoFeatureSelector(AbstractFeatureSelector):
    """
    Lasso Feature Selection.

    Reference: Tibshirani, Robert. "Regression shrinkage and selection via the lasso." Journal of the Royal Statistical Society Series B: Statistical Methodology 58.1 (1996): 267-288.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
    """



    name = "LassoFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        lasso = Lasso(random_state=1)
        lasso.fit(X, y)
        scores = lasso.coef_
        feature_scores = dict(zip(X.columns, scores))
        return feature_scores
