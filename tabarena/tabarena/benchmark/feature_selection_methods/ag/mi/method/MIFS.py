from __future__ import annotations

import logging
import time
import warnings
from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class MIFS:
    """MI feature selector."""

    def __init__(self):
        self._y = None
        self._model = None
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_selected = self.mi(X, y, n_max_features, **kwargs)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def mi(self, X, y, n_max_features, **kwargs):
        """This function implements the MI feature selection.

        Input
        -----
        X: pandas DataFrame,
        y: pandas Series,
        kwargs: {dictionary}

        Output
        ------
        X: pandas DataFrame

        Reference
        ---------
        Battiti, R. (1994). Using mutual information for selecting features in supervised neural net learning. IEEE Transactions on neural networks, 5(4), 537-550. https://doi.org/10.1109/72.298224
        """
        X_np = X.to_numpy()

        _n_samples, n_features = X_np.shape
        MI = np.zeros(n_features)

        t1 = np.zeros(n_features)
        t2 = np.zeros(n_features)
        t3 = np.zeros(n_features)
        for i in range(n_features):
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
            f = X_np[:, i]
            t1[i] = self.entropyd(f)
            t2[i] = self.entropyd(y)
            t3[i] = self.midd(f, y)
            MI[i] = t1[i] + t2[i] - t3[i]

        sorted_idx = np.argsort(-MI)
        selected_features_idx = sorted_idx[:n_max_features]
        selected_features = X.columns[selected_features_idx]
        return X[selected_features]

    def midd(self, x, y):
        """Discrete mutual information estimator given a list of samples which can be any hashable object."""
        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)

    def entropyd(self, sx, base=2):
        """Discrete entropy estimator given a list of samples which can be any hashable object."""
        return self.entropyfromprobs(self.hist(sx), base=base)

    def cmidd(self, x, y, z):
        """Discrete mutual information estimator given a list of samples which can be any hashable object."""
        return (
            self.entropyd(list(zip(y, z)))
            + self.entropyd(list(zip(x, z)))
            - self.entropyd(list(zip(x, y, z)))
            - self.entropyd(z)
        )

    def hist(self, sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return (float(z) / len(sx) for z in d.values())

    def entropyfromprobs(self, probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(self.elog, probs)) / log(base)

    def elog(self, x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0.0 or x >= 1.0:
            return 0
        return x * log(x)
