from __future__ import annotations

import numpy as np
import pandas as pd
import warnings
import logging
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ElasticNetFS:
    """Elastic-net feature selector (classification via LogisticRegression)"""

    def __init__(self, C: float = 1.0, l1_ratio: float = 0.5, max_iter: int = 5000, random_state: int = 0):
        self.C = C
        self.l1_ratio = l1_ratio
        self.max_iter = max_iter
        self.random_state = random_state
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        scores = self.elastic_net_scores(X, y, n_max_features, **kwargs)
        sorted_idx = np.argsort(-scores)
        selected_features_idx = sorted_idx[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def elastic_net_scores(self, X_train: pd.DataFrame, y_train: pd.Series, n_max_features, **kwargs) -> np.ndarray:
        # Time limit check (same pattern as your CART code)
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... "
                    f"(Time Left = {kwargs['time_limit']:.1f}s)"
                )
                score = np.zeros(X_train.shape[1])
                if n_max_features is not None and X_train.shape[1] > n_max_features:
                    selected_idx = np.random.choice(X_train.shape[1], size=n_max_features, replace=False)
                else:
                    selected_idx = np.arange(X_train.shape[1])
                score[selected_idx] = 1
                return score

        clf = make_pipeline(
            StandardScaler(with_mean=True, with_std=True),
            LogisticRegression(
                penalty="elasticnet",
                solver="saga",
                l1_ratio=self.l1_ratio,
                C=self.C,
                max_iter=self.max_iter,
                random_state=self.random_state,
                n_jobs=-1,
            ),
        )
        clf.fit(X_train, y_train)

        # coef_ shape: (n_classes, n_features) or (1, n_features)
        coef = clf.named_steps["logisticregression"].coef_
        scores = np.mean(np.abs(coef), axis=0)  # collapse multiclass to one score per feature

        return scores
