"""Chi-squared feature selection."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


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
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        numeric_imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)

        chi2_kwargs = {"score_func": chi2, "k": "all"}
        chi2FS = SelectKBest(**chi2_kwargs)
        chi2FS.fit(X_imputed, y)
        scores = chi2FS.scores_
        return dict(zip(X.columns, scores))
