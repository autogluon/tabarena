from __future__ import annotations

import time
from math import log
import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class JMI:
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
        This function implements the MI feature selection

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
        Yang, H., & Moody, J. (1999). Feature selection based on joint mutual information. In Proceedings of international ICSC symposium on advances in intelligent data analysis (Vol. 23). Rochester, NY: Citeseer.

        """

        X_np = X.to_numpy()
        y_np = y.to_numpy()

        n_samples, n_features = X_np.shape
        # index of selected features, initialized to be empty
        F = []
        # List of MI's for
        JMI = np.zeros(n_features)

        is_n_selected_features_specified = False
        if self._n_max_features is not None:
            n_selected_features = self._n_max_features
            is_n_selected_features_specified = True
        # select the feature whose j_cmi is the largest
        # t1 stores I(f) for each feature f
        t1 = np.zeros(n_features)
        # t2 stores (I(y) for each feature f
        t2 = np.zeros(n_features)
        # t3 stores I(f|y) for each feature f
        t3 = np.zeros(n_features)
        for i in range(n_features):
            for j in range(n_features):
                if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                    time_start_fit = time.time()
                    kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                    if kwargs["time_limit"] <= 0:
                        logger.warning(
                            f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                        if n_max_features is not None and len(X.columns) > n_max_features:
                            X_out = X.sample(n=n_max_features, axis=1)
                            return X_out
                        else:
                            return X
                f_i_and_j = X_np[:, [i, j]]
                t1[i] = self.entropyd(f_i_and_j)
                t2[i] = self.entropyd(y)
                t3[i] = self.midd(f_i_and_j, y)
                JMI[i] = t1[i] + t2[i] - t3[i]

        j_cmi = 1

        # select the feature whose mutual information is the largest
        while True:
            if "time_limit" in kwargs and kwargs["time_limit"] is not None:
                time_start_fit = time.time()
                kwargs["time_limit"] -= time_start_fit - kwargs["start_time"]
                if kwargs["time_limit"] <= 0:
                    logger.warning(
                        f'\tWarning: FeatureSelection Method has no time left to train... (Time Left = {kwargs["time_limit"]:.1f}s)')
                    if n_max_features is not None and len(X.columns) > n_max_features:
                        X_out = X.sample(n=n_max_features, axis=1)
                        return X_out
                    else:
                        return X
            if len(JMI) == 0:
                idx = np.argmax(JMI)
                F.append(idx)
                f_select = X_np[:, idx]
            if is_n_selected_features_specified:
                if len(F) == n_max_features:
                    break
            else:
                if j_cmi < 0:
                    break
            j_cmi = -1E30

        selected_features_idx = np.array(F)[:n_max_features]
        selected_features = X.columns[selected_features_idx]
        X_selected = X[selected_features]
        return X_selected

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

    def cmidd(self, x, y, z):
        """
        Discrete mutual information estimator given a list of samples which can be any hashable object
        """

        return self.entropyd(list(zip(y, z))) + self.entropyd(list(zip(x, z))) - self.entropyd(
            list(zip(x, y, z))) - self.entropyd(z)

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
