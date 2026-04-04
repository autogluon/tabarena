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

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)
        numeric_imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)

        anova_kwargs = {"score_func": f_classif, "k": "all"}
        anova = SelectKBest(**anova_kwargs)
        anova.fit(X_imputed, y)
        scores = anova.scores_
        return dict(zip(X.columns, scores))
