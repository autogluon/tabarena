from __future__ import annotations

import copy
import logging
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from sklearn.model_selection import train_test_split

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class AccuracyFS:
    """Accuracy feature selector."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        feature_ranking = self.feature_ranking(self.accuracy(X, y, n_max_features, model, **kwargs))

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def accuracy(self, X, y, n_max_features, model, **kwargs):
        """This function calculates the accuracy for each feature.

        Input
        -----
        X: pd.DataFrame, shape (n_samples, n_features)
            input data
        y: pd.Series, shape (n_samples,)
            input class labels

        Output
        ------
        F: {numpy array}, shape (n_features,)
            accuracy for each feature
        """
        F = np.zeros(len(X.columns))
        for i in range(len(X.columns)):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    if n_max_features is not None and len(X.columns) > n_max_features:
                        return X.sample(n=n_max_features, axis=1)
                    return X
            feature_mask = [j != i for j in range(len(X.columns))]
            X_selection = X.iloc[:, feature_mask]
            F[i] = self.evaluate_subset(X_selection, y, model)
        return F

    def evaluate_subset(self, X, y, model):
        X_train, _X_test, y_train, _y_test = train_test_split(X, y, test_size=0.2)
        model_copy = copy.deepcopy(model)
        model_copy.params["fold_fitting_strategy"] = "sequential_local"
        model_copy = model_copy.fit(X=X_train, y=y_train, k_fold=8)
        self._model = model_copy
        return model_copy.score_with_oof(y=y_train)

    def feature_ranking(self, F):
        """Rank features in descending order according to t-score, the higher the t-score, the more important the feature is."""
        idx = np.argsort(F)
        return idx[::-1]
