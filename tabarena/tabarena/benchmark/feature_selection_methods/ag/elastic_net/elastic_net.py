"""Elastic net feature selection."""
from __future__ import annotations

import logging

import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ElasticNetFeatureSelector(AbstractFeatureSelector):
    """ElasticNet Feature Selection.

    Reference: Zou, Hui, and Trevor Hastie. "Regularization and
    variable selection via the elastic net." Journal of the Royal
    Statistical Society Series B: Statistical Methodology 67.2
    (2005): 301-320.
    Implementation Source:
    https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
    """

    name = "ElasticNetFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None,  # noqa: ARG002
    ) -> dict[str, float]:
        data_encoder = OrdinalEncoder()
        X = pd.DataFrame(data_encoder.fit_transform(X), columns=X.columns, index=X.index)
        numeric_imputer = SimpleImputer(strategy="mean")
        X_imputed = pd.DataFrame(numeric_imputer.fit_transform(X), columns=X.columns, index=X.index)
        if self.problem_type == "regression":
            y_processed = y
            C = 1.0
            l1_ratio = 0.5
            max_iter = 5000
            elastic_net = make_pipeline(
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
            elastic_net.fit(X_imputed, y_processed)
            scores = elastic_net.named_steps["logisticregression"].coef_[0]
        else:
            label_encoder = LabelEncoder()
            y_processed = label_encoder.fit_transform(y)
            elastic_net = ElasticNet(random_state=0)
            elastic_net.fit(X_imputed, y_processed)
            scores = elastic_net.coef_
        return dict(zip(X.columns, scores))
