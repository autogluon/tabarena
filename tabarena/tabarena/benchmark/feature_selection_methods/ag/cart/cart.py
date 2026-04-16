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

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(exclude=["object", "category"]).columns.tolist()

        # Numeric: mean imputation; Categorical: most_frequent imputation
        X_imputed = X.copy()
        if num_cols:
            num_imputer = SimpleImputer(strategy="mean")
            X_imputed[num_cols] = num_imputer.fit_transform(X[num_cols])
        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            X_imputed[cat_cols] = cat_imputer.fit_transform(X[cat_cols])

        if cat_cols:
            data_encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            X_imputed[cat_cols] = data_encoder.fit_transform(X_imputed[cat_cols])

        if self.problem_type == "regression":
            y_processed = y
            CART = DecisionTreeRegressor(random_state=self.random_state)
        else:
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y)
            CART = DecisionTreeClassifier(random_state=self.random_state)

        CART.fit(X, y_processed)
        importances = CART.feature_importances_
        return dict(zip(X.columns, importances))
