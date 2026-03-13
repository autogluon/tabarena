import logging

import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class Chi2FeatureSelector(AbstractFeatureSelector):
    """
    Chi2 Feature Selection.

    Reference: Liu, Huan, and Rudy Setiono. "Chi2: Feature selection and discretization of numeric attributes." Proceedings of 7th IEEE international conference on tools with artificial intelligence. Ieee, 1995.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.chi2.html
    """



    name = "Chi2FeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        chi2_kwargs = {"score_func": chi2, "k": "all"}
        chi2FS = SelectKBest(**chi2_kwargs)
        chi2FS.fit(X, y)
        scores = chi2FS.scores_
        feature_scores = dict(zip(X.columns, scores))
        return feature_scores
