from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np
from scipy.stats import entropy

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class InformationGainFS:
    """InformationGain feature selector."""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model=None, n_max_features=None, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
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
        igr_scores = []
        for col in X.columns:
            igr = self._information_gain(X[[col]], col, y, n_max_features, **kwargs)
            igr_scores.append(igr)
        igr_scores = np.array(igr_scores)
        top_features_idx = np.argsort(igr_scores)[-n_max_features:][::-1]
        X_selected = X.iloc[:, top_features_idx]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def _information_gain(self, X: pd.DataFrame, feature: str, y: pd.Series, n_max_features, **kwargs) -> float:
        """Compute information gain for a single feature.

        Parameters
        ----------
        X : pd.DataFrame
            Feature DataFrame
        feature : str
            Column name of the feature
        y : pd.Series
            Target variable

        Returns:
        -------
        float
            Information gain value
        """
        H_y = self._entropy(y.value_counts())
        if "time_limit" in kwargs and kwargs["time_limit"] is not None:
            time_start_fit = time.time()
            kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
            kwargs["start_time"] = time_start_fit
            if kwargs["time_limit"] <= 0:
                logger.warning(
                    f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                )
                score = np.zeros(X.shape[1])
                if n_max_features is not None and X.shape[1] > n_max_features:
                    selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                else:
                    selected_idx = np.arange(X.shape[1])
                score[selected_idx] = 1
                return score
        subset_entropy = 0.0
        for value in X[feature].unique():
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    score = np.zeros(X.shape[1])
                    if n_max_features is not None and X.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X.shape[1])
                    score[selected_idx] = 1
                    return score
            mask = X[feature] == value
            y_subset = y[mask]
            if len(y_subset) > 0:
                # Calculate entropy of this subset
                subset_entropy = self._entropy(y_subset.value_counts())
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f"\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs['time_limit']:.1f}s)"
                        )
                        score = np.zeros(X.shape[1])
                        if n_max_features is not None and X.shape[1] > n_max_features:
                            selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                        else:
                            selected_idx = np.arange(X.shape[1])
                        score[selected_idx] = 1
                        return score
        # Information gain = H(Y) - H(Y|feature)
        return H_y - subset_entropy

    @staticmethod
    def _entropy(values):
        """Calculate entropy from a frequency distribution.

        Parameters
        ----------
        values : array-like
            Value counts or frequencies

        Returns:
        -------
        float
            Entropy value
        """
        # Normalize to probabilities and compute entropy
        probabilities = values / values.sum()
        return entropy(probabilities, base=2)
