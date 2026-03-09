from __future__ import annotations

import time
import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class JMIFS:
    """JMI feature selector"""

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_selected = self.jmi(X, y, n_max_features, **kwargs)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def jmi(self, X, y, n_max_features, **kwargs):
        """
        Implements Joint Mutual Information (JMI) feature selection using
        the Kullback-Leibler divergence definition:

            I(X_1,...,X_k ; Y) = KL( p(x1,...,xk, y) || p(x1,...,xk) * p(y) )

        Input
        -----
        X : pandas DataFrame
        y : pandas Series
        n_max_features : int
        kwargs : dict (supports 'time_limit' and 'start_time')

        Output
        ------
        X_selected : pandas DataFrame

        Reference
        ---------
        Yang, H., & Moody, J. (1999). Data visualization and feature selection: New algorithms for nongaussian data.
        Advances in neural information processing systems, 12.
        """

        X_np = self._discretize(X.to_numpy(), n_max_features, **kwargs)
        y_np = y.to_numpy()

        n_samples, n_features = X_np.shape
        selected = []  # indices of selected features
        remaining = list(range(n_features))

        first_scores = np.zeros(n_features)
        for i in remaining:
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    if n_max_features is not None and len(X.columns) > n_max_features:
                        X_out = X.sample(n=n_max_features, axis=1)
                        return X_out
                    else:
                        return X
            first_scores[i] = self._joint_mi_kl(X_np[:, [i]], y_np, n_max_features, **kwargs)

        best_first = int(np.argmax(first_scores))
        selected.append(best_first)
        remaining.remove(best_first)

        while len(selected) < n_max_features and remaining:
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    if n_max_features is not None and len(X.columns) > n_max_features:
                        X_out = X.sample(n=n_max_features, axis=1)
                        return X_out
                    else:
                        return X
            best_score = -np.inf
            best_idx = None
            for i in remaining:
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                        if n_max_features is not None and len(X.columns) > n_max_features:
                            X_out = X.sample(n=n_max_features, axis=1)
                            return X_out
                        else:
                            return X
                candidate_cols = selected + [i]
                X_subset = X_np[:, candidate_cols]

                score = self._joint_mi_kl(X_subset, y_np, n_max_features, **kwargs)

                if score > best_score:
                    best_score = score
                    best_idx = i
            selected.append(best_idx)
            remaining.remove(best_idx)
        selected_features = X.columns[selected]
        return X[selected_features]

    @staticmethod
    def _discretize(X_np: np.ndarray, n_max_features, n_bins: int = 10, **kwargs) -> np.ndarray:
        """Bin continuous features into integers for probability estimation."""
        X_disc = np.zeros_like(X_np, dtype=int)
        for i in range(X_np.shape[1]):
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
            col = X_np[:, i]
            bins = np.linspace(col.min(), col.max(), n_bins)
            X_disc[:, i] = np.digitize(col, bins)
        return X_disc

    @staticmethod
    def _estimate_prob(data: np.ndarray, n_max_features,  **kwargs) -> dict:
        """Estimate joint probability distribution from rows of data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = data.shape[0]
        counts = {}
        for row in data:
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    score = np.zeros(data.shape[1])
                    if n_max_features is not None and data.shape[1] > n_max_features:
                        selected_idx = np.random.choice(data.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(data.shape[1])
                    score[selected_idx] = 1
                    return score
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        return {k: v / n for k, v in counts.items()}

    def _joint_mi_kl(self, X_subset: np.ndarray, y_np: np.ndarray, n_max_features, **kwargs) -> float:
        """
        I(X_1,...,X_k ; Y) = KL( p(x,y) || p(x)*p(y) )
                           = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )
        """
        if X_subset.ndim == 1:
            X_subset = X_subset.reshape(-1, 1)

        y_col = y_np.reshape(-1, 1)

        p_xy = self._estimate_prob(np.hstack([X_subset, y_col]), n_max_features, **kwargs)
        p_x = self._estimate_prob(X_subset, n_max_features, **kwargs)
        p_y = self._estimate_prob(y_col, n_max_features, **kwargs)

        jmi = 0.0
        for xy_key, p_xy_val in p_xy.items():
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    score = np.zeros(X_subset.shape[1])
                    if n_max_features is not None and X_subset.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X_subset.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X_subset.shape[1])
                    score[selected_idx] = 1
                    return score
            x_key = xy_key[:-1]
            y_key = (xy_key[-1],)

            p_x_val = p_x.get(x_key, 0)
            p_y_val = p_y.get(y_key, 0)

            if p_x_val > 0 and p_y_val > 0:
                jmi += p_xy_val * np.log(p_xy_val / (p_x_val * p_y_val))

        return jmi
