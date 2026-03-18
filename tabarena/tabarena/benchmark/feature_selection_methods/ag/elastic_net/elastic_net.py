from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ElasticNetFeatureSelector(AbstractFeatureSelector):
    """ElasticNet Feature Selection.

    Reference: Zou, Hui, and Trevor Hastie. "Regularization and variable selection via the elastic net." Journal of the Royal Statistical Society Series B: Statistical Methodology 67.2 (2005): 301-320.
    Implementation Source: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    name = "ElasticNetFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns, index=X.index)
        C = 1.0
        l1_ratio = 0.5
        max_iter = 5000
        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=l1_ratio,
                C=C,
                max_iter=max_iter,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        )
        clf.fit(X_imputed, y)
        scores = clf.named_steps["logisticregression"].coef_[0]
        return dict(zip(X.columns, scores))
