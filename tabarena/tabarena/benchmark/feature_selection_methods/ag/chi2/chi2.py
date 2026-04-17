"""Chi-squared feature selection."""
from __future__ import annotations

import pandas as pd
import numpy as np

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, KBinsDiscretizer

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

class Chi2FeatureSelector(AbstractFeatureSelector):
    """Chi2 Feature Selection.

    Reference: Liu, Huan, and Rudy Setiono. "Chi2: Feature selection
    and discretization of numeric attributes." Proceedings of 7th IEEE
    international conference on tools with artificial intelligence.
    Ieee, 1995.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    """

    name = "Chi2FeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        if self.problem_type == "regression":
            raise ValueError("Chi² is not applicable to regression tasks.")

        cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()
        num_cols = X.select_dtypes(include="number").columns.tolist()
        
        X_imputed = X.copy()
        if num_cols:
            X_imputed[num_cols] = SimpleImputer(strategy="mean").fit_transform(X_imputed[num_cols])
            X_imputed = self._discretize(X_imputed) # chi2 needs cate
        if cat_cols: #impute and encode
            X_imputed[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(X_imputed[cat_cols])
            X_imputed[cat_cols] = OrdinalEncoder(
                handle_unknown="use_encoded_value", 
                unknown_value=0
            ).fit_transform(X_imputed[cat_cols])

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        chi2_kwargs = {"score_func": chi2, "k": "all"}
        chi2FS = SelectKBest(**chi2_kwargs)
        chi2FS.fit(X_imputed, y)
        scores = np.nan_to_num(chi2FS.scores_, nan=0.0)
        return dict(zip(X.columns, scores))

    def _discretize(self, X: pd.DataFrame) -> pd.DataFrame:
        """Discretize continuous attributes with unsupervised method (original paper refers to Fayyad & Irani MDL)"""
        X_disc = X.copy()
        num_cols = X.select_dtypes(include="number").columns
        max_bins = int(np.log2(len(X))) + 1  # sturges' rule to scale with size
        
        for col in num_cols:
            n_bins = min(X[col].nunique(), max_bins)
            if n_bins < 2:
                continue  # constant feature, skip
            X_disc[col] = KBinsDiscretizer(
                n_bins=n_bins, encode="ordinal", strategy="quantile"
            ).fit_transform(X[[col]])
        return X_disc