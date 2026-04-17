"""CART (decision tree) feature selection."""
from __future__ import annotations

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


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
        num_cols = X.select_dtypes(include="number").columns.tolist()

        X_imputed = X.copy()
        if num_cols:
            X_imputed[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
        if cat_cols:
            X_imputed[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
            X_imputed[cat_cols] = OrdinalEncoder(
                handle_unknown="use_encoded_value", 
                unknown_value=-1
            ).fit_transform(X_imputed[cat_cols])

        if self.problem_type == "regression":
            y_processed = y
            CART = DecisionTreeRegressor(random_state=self.random_state)
        else:
            y_processed = LabelEncoder().fit_transform(y)
            CART = DecisionTreeClassifier(random_state=self.random_state)

        CART.fit(X_imputed, y_processed)
        importances = CART.feature_importances_
        return dict(zip(X.columns, importances))
