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


class GainRatioFS:
    """GainRatio feature selector."""

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

        scores = self.gain_ratio(X, y, n_max_features, **kwargs)
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

    def gain_ratio(self, X: pd.DataFrame, y: pd.Series, n_max_features, **kwargs) -> np.ndarray:
        """Gain Ratio for each feature:
          GR = InformationGain(X_i, Y) / SplitInfo(X_i).

        X: DataFrame (n_samples, n_features)
        y: Series   (n_samples,)
        """
        _n_samples, n_features = X.shape
        F = np.zeros(n_features, dtype=float)

        # Parent entropy H(Y)
        e_parent = self._entropy_from_counts(y.value_counts(dropna=False))  # [web:19][web:27]

        for i in range(n_features):
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
            f = X.iloc[:, i]

            # SplitInfo(X_i) = - sum_v p(v) log2 p(v)
            p_v = f.value_counts(normalize=True, dropna=False)  # probabilities [web:19]
            split_info = -(p_v * np.log2(p_v)).sum()

            # Conditional entropy H(Y | X_i) = sum_v p(v) H(Y | X_i=v)
            e_child = 0.0
            for v, p in p_v.items():
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
                y_sub = y[f.eq(v)]
                e_child += p * self._entropy_from_counts(y_sub.value_counts(dropna=False))  # [web:19][web:27]

            info_gain = e_parent - e_child

            F[i] = info_gain / split_info if split_info > 0 else 0.0

        return np.abs(F)

    def feature_ranking(self, F: np.ndarray) -> np.ndarray:
        """Rank features in descending order (higher gain ratio is better)."""
        return np.argsort(F)[::-1]

    @staticmethod
    def _entropy_from_counts(counts: pd.Series) -> float:
        """Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts. [web:27].
        """
        return float(entropy(counts.to_numpy(), base=2))  # base=2 => bits [web:27]
