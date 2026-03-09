from __future__ import annotations

import time
from math import log

import numpy as np
import pandas as pd

import warnings
import logging

logger = logging.getLogger(__name__)
warnings.filterwarnings('ignore')


class DISRFS:
    """ DISR feature selector """

    def __init__(self, model):
        self._y = None
        self._model = model
        self._n_max_features = None
        self._selected_features = None

    def fit_transform(self, X: pd.DataFrame, y: pd.Series, model, n_max_features, **kwargs) -> pd.DataFrame:
        self._y = y
        self._model = model
        self._n_max_features = n_max_features
        X_selected = self.disr(X, y, n_max_features, **kwargs)
        self._selected_features = list(X_selected.columns)
        return X_selected

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._selected_features is None:
            self.fit_transform(X, self._y, self._model, self._n_max_features)
        return X[self._selected_features]

    def disr(self, X, y, n_max_features, **kwargs):
        """
        This function implement the DISR feature selection.
        The scoring criteria is calculated based on the formula j_disr=sum_j(I(f,fj;y)/H(f,fj,y))

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
        Meyer, P. E., & Bontempi, G. (2006, April). On the use of variable complementarity for feature selection in cancer classification. In Workshops on applications of evolutionary computation (pp. 91-102). Berlin, Heidelberg: Springer Berlin Heidelberg.
        """

        X_np = X.to_numpy()

        n_samples, n_features = X_np.shape
        DISR = np.zeros(n_features)

        mutual_information = np.zeros(n_features)
        entropy = np.zeros(n_features)
        for i in range(n_features):
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
            f = X_np[:, i]
            mutual_information[i] = self.midd(f, y)
            entropy[i] = self.entropyd(list(zip(f, y)))
            symmetrical_relevance = mutual_information[i] / entropy[i]
            DISR[i] = symmetrical_relevance

        sorted_idx = np.argsort(-DISR)
        selected_features_idx = sorted_idx[:n_max_features]
        selected_features = X.columns[selected_features_idx]
        X_selected = X[selected_features]
        return X_selected


    def conditional_entropy(self, f1, f2):
        """
        This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)

        Input
        -----
        f1: {numpy array}, shape (n_samples,)
        f2: {numpy array}, shape (n_samples,)

        Output
        ------
        ce: {float}
            ce is conditional entropy of f1 and f2
        """
        ce = self.entropyd(f1) - self.midd(f1, f2)
        return ce

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
        return self.entropyd(list(zip(y, z))) + self.entropyd(list(zip(x, z))) - self.entropyd(list(zip(x, y, z))) - self.entropyd(z)

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