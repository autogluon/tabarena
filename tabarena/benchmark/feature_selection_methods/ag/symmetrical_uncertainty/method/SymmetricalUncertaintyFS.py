from __future__ import annotations

import time

import numpy as np
import pandas as pd
import warnings
import logging

from scipy.stats import entropy

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class SymmetricalUncertaintyFS:
    """SymmetricalUncertainty feature selector"""

    def __init__(self, model=None):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model=None,
        n_max_features: int | None = None,
        **kwargs,
    ) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        scores = self.symmetrical_uncertainty(X, y, n_max_features, **kwargs)
        feature_ranking = self.feature_ranking(scores)

        if n_max_features is None or n_max_features >= X.shape[1]:
            selected_features = X.columns[feature_ranking]
        else:
            selected_features = X.columns[feature_ranking[:n_max_features]]

        X_selected = X.loc[:, selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            # fall back to stored params
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X.loc[:, self._selected_features]

    def symmetrical_uncertainty(self, X: pd.DataFrame, y: pd.Series, n_max_features, **kwargs
    ) -> np.ndarray:
        """
        Symmetrical Uncertainty for each feature:
          SU(X, Y) = 2 * IG(Y|X) / (H(X) + H(Y))
        where:
          IG(Y|X) = H(Y) - H(Y|X)  (information gain / mutual information). [web:39][web:31]

        Returns
        -------
        np.ndarray of shape (n_features,)
        """
        n_samples, n_features = X.shape
        F = np.zeros(n_features, dtype=float)

        # H(Y)
        H_y = self._entropy_from_counts(y.value_counts(dropna=False))

        for i in range(n_features):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... "
                        f"(Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    score = np.zeros(X.shape[1])
                    if n_max_features is not None and X.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                    else:
                        selected_idx = np.arange(X.shape[1])
                    score[selected_idx] = 1
                    return score

            f = X.iloc[:, i]
            # H(X)
            H_x = self._entropy_from_counts(f.value_counts(dropna=False))
            # H(Y|X) = sum_v p(v) * H(Y | X=v)
            p_v = f.value_counts(normalize=True, dropna=False)
            H_y_given_x = 0.0
            for v, p in p_v.items():
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f"\tWarning: FeatureSelection Method has no time left to train... "
                            f"(Time Left = {kwargs['time_limit']:.1f}s)"
                        )
                        score = np.zeros(X.shape[1])
                        if n_max_features is not None and X.shape[1] > n_max_features:
                            selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                        else:
                            selected_idx = np.arange(X.shape[1])
                        score[selected_idx] = 1
                        return score

                y_sub = y[f.eq(v)]
                H_y_given_x += p * self._entropy_from_counts(y_sub.value_counts(dropna=False))  # [web:19][web:27]

            IG = H_y - H_y_given_x  # IG(Y|X) = H(Y) - H(Y|X)

            denom = H_x + H_y
            F[i] = (2.0 * IG / denom) if denom > 0 else 0.0  # SU definition

        return np.abs(F)

    def feature_ranking(self, F: np.ndarray) -> np.ndarray:
        """Rank features in descending order (higher gain ratio is better)."""
        return np.argsort(F)[::-1]

    @staticmethod
    def _entropy_from_counts(counts: pd.Series) -> float:
        """
        Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts.
        """
        return float(entropy(counts.to_numpy(), base=2))  # base=2 => bits
