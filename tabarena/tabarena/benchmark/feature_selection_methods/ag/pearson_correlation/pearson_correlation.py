from __future__ import annotations

import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, r_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class PearsonCorrelationFeatureSelector(AbstractFeatureSelector):
    """PearsonCorrelation Feature Selection.

    Reference: Liu, Huan, and Rudy Setiono. "Chi2: Feature selection and discretization of numeric attributes." Proceedings of 7th IEEE international conference on tools with artificial intelligence. Ieee, 1995.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    """

    name = "PearsonCorrelationFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        numeric_imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)

        pc_kwargs = {"score_func": r_regression, "k": "all"}
        pcFS = SelectKBest(**pc_kwargs)
        pcFS.fit(X_imputed, y)
        scores = pcFS.scores_
        return dict(zip(X.columns, scores))
