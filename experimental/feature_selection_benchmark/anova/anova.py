import logging

import pandas as pd
from sklearn.feature_selection import f_classif, SelectKBest

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ANOVAFeatureSelector(AbstractFeatureSelector):
    """ANOVA Feature Selection."""

    name = "ANOVAFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        anova_kwargs = {"score_func": f_classif, "k": "all"}
        anova = SelectKBest(**anova_kwargs)
        anova.fit(X, y)
        scores = anova.scores_
        feature_scores = dict(zip(X.columns, scores))
        return feature_scores
