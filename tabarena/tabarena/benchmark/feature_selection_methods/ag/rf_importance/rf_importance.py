from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.ensemble import RandomForestRegressor

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class RFImportanceFeatureSelector(AbstractFeatureSelector):
    """RFImportance Feature Selection.

    Reference: Breiman, Leo. "Random forests." Machine learning 45.1 (2001): 5-32.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
    """

    name = "RFImportanceFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        forest = RandomForestRegressor(random_state=0)
        forest.fit(X, y)
        importances = forest.feature_importances_
        return dict(zip(X.columns, importances))
