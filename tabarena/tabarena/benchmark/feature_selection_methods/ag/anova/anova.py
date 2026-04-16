"""ANOVA feature selection."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ANOVAFeatureSelector(AbstractFeatureSelector):
    """ANOVA Feature Selection.

    Reference: St, Lars, and Svante Wold. "Analysis of variance
    (ANOVA)." Chemometrics and intelligent laboratory systems
    6.4 (1989): 259-272.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
    """

    name = "ANOVAFeatureSelector"
    feature_scoring_method: bool = True

    # TODO: split for regression/classification
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

        anova_kwargs = {"score_func": f_classif, "k": "all"}
        anova = SelectKBest(**anova_kwargs)
        anova.fit(X_imputed, y)
        scores = anova.scores_
        return dict(zip(X.columns, scores))
