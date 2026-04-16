"""CART (decision tree) feature selection."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class CARTFeatureSelector(AbstractFeatureSelector):
    """CART Feature Selection.

    Reference: Breiman, Leo, et al. Classification and regression trees. Chapman and Hall/CRC, 2017.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
    """

    name = "CARTFeatureSelector"
    feature_scoring_method: bool = True

    # TODO: encode only categorical
    # TODO: split imputation based on attr type
    # TODO: use random state attribute from AbstractFeatureSelector
    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        numeric_imputer = SimpleImputer(strategy="most_frequent") # handles categorical feats too
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X_imputed), columns=X_imputed.columns, index=X_imputed.index)
        
        if self.problem_type == "regression":
            y_processed = y
            CART = DecisionTreeRegressor(random_state=0)
        else:
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y)
            CART = DecisionTreeClassifier(random_state=0)

        CART.fit(X, y_processed)
        importances = CART.feature_importances_
        return dict(zip(X.columns, importances))
