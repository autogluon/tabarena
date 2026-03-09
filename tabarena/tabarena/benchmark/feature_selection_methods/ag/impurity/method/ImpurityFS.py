from __future__ import annotations

import time

import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class ImpurityFS:
    """Impurity feature selector"""

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        feature_ranking = self.feature_ranking(self.impurity(X_np, y_np, n_max_features, **kwargs))

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def impurity(self, X_np: np.ndarray, y_np: np.ndarray, n_max_features, **kwargs) -> np.ndarray:
        n_features = X_np.shape[1]
        alpha = np.zeros(n_features)
        for w in range(n_features):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    score = np.zeros(X_np.shape[1])
                    if n_max_features is not None and X_np.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X_np.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X_np.shape[1])
                    score[selected_idx] = 1
                    return score
            # Sort the dataset by feature w
            sorted_indices = np.argsort(X_np[:, w])
            sorted_labels = y_np[sorted_indices]
            alpha[w] = self.classification_error_impurity(X_np, sorted_labels, n_max_features, **kwargs)
        return alpha

    @staticmethod
    def classification_error_impurity(X_np, arr, n_max_features, **kwargs):
        unique_elements = np.unique(arr)
        w_values = []
        for current_element in unique_elements:
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    score = np.zeros(X_np.shape[1])
                    if n_max_features is not None and X_np.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X_np.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X_np.shape[1])
                    score[selected_idx] = 1
                    return score
            if current_element not in arr:
                continue
            # Find indices of the current element in the array
            indices = np.where(arr == current_element)[0]
            # Create subarray for the current element
            arr_element = arr[indices[0]:indices[-1] + 1]
            # Count the occurrences of other elements in the subarray
            n_other = np.count_nonzero(arr_element != current_element)
            # Count the occurrences of the current element in the subarray
            n_current = np.count_nonzero(arr_element == current_element)
            # Avoid division by zero
            if n_current == 0:
                w_values.append(float('inf'))  # or any other suitable value
            else:
                w = n_other / (n_current + n_other)
                w_values.append(w)
        return 1 - min(w_values)

    @staticmethod
    def feature_ranking(F):
        """
        Rank features in descending order according to impurity, the higher the impurity, the less important the feature is
        """
        idx = np.argsort(-F)
        return idx[::-1]
