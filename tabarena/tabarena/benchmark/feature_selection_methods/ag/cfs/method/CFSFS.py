from __future__ import annotations

import logging
import warnings
from math import log
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class CFSFS:
    """CFS feature selector.

    There exist the following variation: CFS-UC uses symmetrical uncertainty to measure
    correlations, CFS-MDL uses normalized symmetrical MDL to measure correlations, and
    CFS-Relief uses symmetrical relief to measure correlations

    Here, we used the CFS-UC implementation from https://github.com/ZixiaoShen/Correlation-based-Feature-Selection
    """

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

        feature_ranking = self.cfs(X_np, y_np)

        selected_features_idx = feature_ranking[:n_max_features]
        selected_features = X.columns[selected_features_idx]

        X_selected = X[selected_features]
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def cfs(self, X, y):
        """This function uses a correlation based heuristic to evaluate the worth of features which is called CFS.

        :param X: {numpy array}, shape (n_samples, n_features) input data
        :param y: {numpy array}, shape (n_samples) input class labels
        :return F: {numpy array}, index of selected features
        """
        _n_samples, n_features = X.shape
        F = []
        M = []  # M stores the merit values
        while True:
            merit = -100000000000
            idx = -1
            for i in range(n_features):
                if i not in F:
                    F.append(i)
                    # calculate the merit of current selected features
                    t = self.merit_calculation(X[:, F], y)
                    if t > merit:
                        merit = t
                        idx = i
                    F.pop()
            F.append(idx)
            M.append(merit)
            if len(M) > 5 and M[len(M) - 1] <= M[len(M) - 2] and M[len(M) - 2] <= M[len(M) - 3]:
                if M[len(M) - 3] <= M[len(M) - 4] and M[len(M) - 4] <= M[len(M) - 5]:
                    break
        return np.array(F)

    def merit_calculation(self, X, y):
        """This function calculates the merit of X given class labels y, where
        merits = (k * rcf) / sqrt (k + k*(k-1)*rff)
        rcf = (1/k)*sum(su(fi, y)) for all fi in X
        rff = (1/(k*(k-1)))*sum(su(fi, fj)) for all fi and fj in X.

        :param X:  {numpy array}, shape (n_samples, n_features) input data
        :param y:  {numpy array}, shape (n_samples) input class labels
        :return merits: {float}  merit of a feature subset X
        """
        _n_samples, n_features = X.shape
        rff = 0
        rcf = 0
        for i in range(n_features):
            fi = X[:, i]
            rcf += self.su_calculation(fi, y)  # su is the symmetrical uncertainty of fi and y
            for j in range(n_features):
                if j > i:
                    fj = X[:, j]
                    rff += self.su_calculation(fi, fj)
        rff *= 2
        return rcf / np.sqrt(n_features + rff)

    def information_gain(self, f1, f2):
        r"""This function calculates the information gain, where ig(f1, f2) = H(f1) - H(f1\f2).

        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: ig: {float}
        """
        return self.entropyd(f1) - self.conditional_entropy(f1, f2)

    def conditional_entropy(self, f1, f2):
        """This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: ce {float} conditional entropy of f1 and f2.
        """
        return self.entropyd(f1) - self.midd(f1, f2)

    def su_calculation(self, f1, f2):
        """This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: su {float} su is the symmetrical uncertainty of f1 and f2.
        """
        # calculate information gain of f1 and f2, t1 = ig(f1, f2)
        t1 = self.information_gain(f1, f2)
        # calculate entropy of f1
        t2 = self.entropyd(f1)
        # calculate entropy of f2
        t3 = self.entropyd(f2)

        return 2.0 * t1 / (t2 + t3)

    def entropyd(self, sx, base=2):
        """Discrete entropy estimator given a list of samples which can be any hashable object."""
        return self.entropyfromprobs(self.hist(sx), base=base)

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

    def midd(self, x, y):
        """Discrete mutual information estimator given a list of samples which can be any hashable object."""
        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)
