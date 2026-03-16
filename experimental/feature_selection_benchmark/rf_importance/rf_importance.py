import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class RFImportanceFeatureSelector(AbstractFeatureSelector):
    """
    RFImportance Feature Selection.

    Reference: Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    name = "RFImportanceFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        c = np.unique(y.to_numpy())
        if c < 10:
            forest = RandomForestClassifier(random_state=0)
        else:
            forest = RandomForestRegressor(random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        feature_scores = dict(zip(X.columns, importances))
        return feature_scores
