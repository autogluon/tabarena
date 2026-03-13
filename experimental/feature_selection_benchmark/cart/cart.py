import logging

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class CARTFeatureSelector(AbstractFeatureSelector):
    """
    CART Feature Selection.

    Reference: Breiman, Leo, et al. Classification and regression trees. Chapman and Hall/CRC, 2017.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    name = "CARTFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        CART = DecisionTreeClassifier(random_state=0)
        CART.fit(X, y)
        importances = CART.feature_importances_
        feature_scores = dict(zip(X.columns, importances))
        return feature_scores
