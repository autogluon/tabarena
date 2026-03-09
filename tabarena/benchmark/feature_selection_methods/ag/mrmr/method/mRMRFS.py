from __future__ import annotations

import time
from math import log
import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class mRMRFS:
    """mRMR feature selector"""

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_selected = self.mrmr(X, y, n_max_features, **kwargs)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]


    def mrmr(self, X, y, n_max_features, **kwargs):
        """

        Reference
        ---------
        Peng, H., Long, F., & Ding, C. (2005). Feature selection based on mutual information criteria of max-dependency, max-relevance, and min-redundancy. IEEE Transactions on pattern analysis and machine intelligence, 27(8), 1226-1238.
        """
        n_features = len(X.columns)
        relevance = np.zeros(n_features)
        redundancy = np.zeros(n_features)
        mRMR_score = np.zeros(n_features)
        for i in range(n_features):
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                kwargs["start_time"] = time_start_fit
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    if n_max_features is not None and X.shape[1] > n_max_features:
                        selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                        X_out = X[:, selected_idx]
                        return X_out
                    else:
                        return X
            f = X[:, i]
            relevance[i] = self.midd(f, y)
            for j in range(n_features):
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    kwargs["start_time"] = time_start_fit
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                        if n_max_features is not None and X.shape[1] > n_max_features:
                            selected_idx = np.random.choice(X.shape[1], size=n_max_features, replace=False)
                            X_out = X[:, selected_idx]
                            return X_out
                        else:
                            return X
                if j != i:
                    f_j = X[:, j]
                    redundancy[i] += self.midd(f_j, f)
            mRMR_score[i] += relevance - redundancy

        if n_max_features is not None and n_features > n_max_features:
            selected_idx = np.argsort(mRMR_score)[::-1][:n_max_features]
        else:
            selected_idx = np.argsort(mRMR_score)[::-1]

        # Return DataFrame with selected columns
        return X.iloc[:, selected_idx]

    def midd(self, x, y):
        """
        Discrete mutual information estimator given a list of samples which can be any hashable object
        """
        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)

    def entropyd(self, sx, base=2):
        """
        Discrete entropy estimator given a list of samples which can be any hashable object
        """
        return self.entropyfromprobs(self.hist(sx), base=base)

    def hist(self, sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return map(lambda z: float(z) / len(sx), d.values())

    def entropyfromprobs(self, probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(self.elog, probs)) / log(base)

    def elog(self, x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0. or x >= 1.:
            return 0
        else:
            return x * log(x)
