from __future__ import annotations

import logging
import time
import warnings
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class ConsistencyFS:
    """Consistency feature selector.

    Reference:
    Liu, H., & Setiono, R. (1996, July). A probabilistic approach to feature selection-a filter solution. In ICML (Vol. 96, pp. 319-327).
    """

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series, model=None, n_max_features=None, r: int = 77, theta: float = 0.0, **kwargs
    ) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features

        selected_idx = self.inconsistency(X=X, y=y, r=r, theta=theta, n_max_features=n_max_features, **kwargs)
        X_out = X.iloc[:, selected_idx]
        self._selected_features = list(X_out.columns)
        return X_out

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            raise RuntimeError("Call fit_transform before transform.")
        return X.loc[:, self._selected_features]

    def inconsistency(
        self, X: pd.DataFrame, y: pd.Series, r: int, theta: float, n_max_features, **kwargs
    ) -> np.ndarray:
        _n_samples, n_features = X.shape
        rng = np.random.default_rng(1)

        c_best = n_features
        s_best = np.ones(n_features, dtype=bool)

        for _ in range(r):
            # Time limit handling (same pattern as your code)
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... "
                        f"(Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    return np.where(s_best)[0]

            # 4: S = randomSet(seed)  (ensure at least 1 feature)
            S = rng.random(n_features) < 0.5
            if not S.any():
                S[rng.integers(0, n_features)] = True

            # Optional: enforce an upper bound on subset size if you still want n_max_features
            if n_max_features is not None and S.sum() > n_max_features:
                on = np.where(S)[0]
                keep = rng.choice(on, size=n_max_features, replace=False)
                S[:] = False
                S[keep] = True

            C = int(S.sum())  # 5: numOfFeatures(S)
            if c_best < C:
                continue  # can't beat current best size

            IR = self._inconsistency_rate(X.loc[:, X.columns[S]], y, n_max_features, **kwargs)  # 7/11
            if IR is None:
                return np.where(s_best)[0]
            # 6–13: accept if IR < theta and (smaller subset or same size)
            if theta > IR and (c_best > C or (c_best == C)):
                c_best = C
                s_best = S.copy()

        return np.where(s_best)[0]

    @staticmethod
    def _inconsistency_rate(X_sub: pd.DataFrame, y: pd.Series, n_max_features, **kwargs) -> float:
        """IR(S) = (sum over patterns) (|pattern_group| - max_class_count) / n."""
        if X_sub.shape[1] == 0:
            return 1.0  # empty set cannot discriminate at all

        df = X_sub.copy()
        df["_y_"] = y.to_numpy()

        incons = 0
        for _, grp in df.groupby(list(X_sub.columns), dropna=False):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f"\tWarning: FeatureSelection Method has no time left to train... "
                        f"(Time Left = {kwargs['time_limit']:.1f}s)"
                    )
                    return None
            counts = grp["_y_"].value_counts(dropna=False)
            incons += len(grp) - int(counts.max())

        return incons / len(df)
