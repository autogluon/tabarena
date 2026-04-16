"""ANOVA feature selection."""
from __future__ import annotations

import numpy as np
import pandas as pd

from sklearn.feature_selection import SelectKBest, f_classif, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector


class ANOVAFeatureSelector(AbstractFeatureSelector):
    """ANOVA Feature Selection.

    Reference: St, Lars, and Svante Wold. "Analysis of variance(ANOVA)." 
    Chemometrics and intelligent laboratory systems 6.4 (1989): 259-272.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.f_classif.html
    """

    name = "ANOVAFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002 - non-iterative method
    ) -> dict[str, float]:
        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include="number").columns.tolist()

        X_imputed = X.copy()
        if num_cols:
            X_imputed[num_cols] = SimpleImputer(strategy="mean").fit_transform(X[num_cols])
        if cat_cols: #impute and encode
            # TODO: think of a better way as ordinal encoding is not valid for ANOVA
            X_imputed[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X[cat_cols])
            X_imputed[cat_cols] = OrdinalEncoder(
                handle_unknown="use_encoded_value", 
                unknown_value=-1
            ).fit_transform(X_imputed[cat_cols])

        if self.problem_type == "regression":
            score_func = f_regression
        
        else:
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            score_func = f_classif

        anova = SelectKBest(score_func=score_func, k="all")
        anova.fit(X_imputed, y)
        scores = np.nan_to_num(anova.scores_, nan=0.0) # in case of constant features that return nan

        return dict(zip(X.columns, scores))