from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from sklearn.tree import DecisionTreeClassifier

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class CARTFeatureSelector(AbstractFeatureSelector):
    """CART Feature Selection.

    Reference: Breiman, Leo, et al. Classification and regression trees. Chapman and Hall/CRC, 2017.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    name = "CARTFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        CART = DecisionTreeClassifier(random_state=0)
        CART.fit(X, y)
        importances = CART.feature_importances_
        return dict(zip(X.columns, importances))
