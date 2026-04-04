"""Random forest importance feature selection."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class RFImportanceFeatureSelector(AbstractFeatureSelector):
    """RFImportance Feature Selection.

    Reference: Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    name = "RFImportanceFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,
    ) -> dict[str, float]:
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        forest = RandomForestRegressor(random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        return dict(zip(X.columns, importances))
