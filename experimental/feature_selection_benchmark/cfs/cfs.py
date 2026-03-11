import logging
import time
from math import log

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class CFSFeatureSelector(AbstractFeatureSelector):
    """
    CFS Feature Selection.

    Reference: Hall, Mark A. Correlation-based feature selection for machine learning. Diss. The University of Waikato, 1999.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/statistical_based/CFS.py#L40.
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
                           This particular implementation of the repo is based on http://featureselection.asu.edu, which for the CFS algorithm cites Hall, Mark A., and Lloyd A. Smith. "Feature selection for machine learning: comparing a correlation-based filter approach to the wrapper." Proceedings of the twelfth international Florida artificial intelligence research society conference. 1999.
                           The variation implemented here is a forward selection method using Symmetrical Uncertainty.
    Changes to the implementation:
                           - Remove early stopping
                           - Break loop when max_features is reached
    """

    name = "CFSFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        columns = X.columns
        start_time = time.monotonic()
        X = X.to_numpy()
        y = y.to_numpy()
        n_samples, n_features = X.shape
        F = []
        M = []  # M stores the merit values
        while True:
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            merit = -100000000000
            idx = -1
            for i in range(n_features):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
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
            if len(M) > 5:
                if M[len(M) - 1] <= M[len(M) - 2]:
                    if M[len(M) - 2] <= M[len(M) - 3]:
                        if M[len(M) - 3] <= M[len(M) - 4]:
                            if M[len(M) - 4] <= M[len(M) - 5]:
                                break
            if len(F) >= self.max_features:
                break
        selected_features = [self._original_features[i] for i in np.array(F)]
        selected_features = selected_features[:self.max_features]
        return [str(feat) for feat in selected_features]
        #feature_scores = dict(zip(columns, M))
        #return feature_scores

    def merit_calculation(self, X, y):
        """
        This function calculates the merit of X given class labels y, where
        merits = (k * rcf) / sqrt (k + k*(k-1)*rff)
        rcf = (1/k)*sum(su(fi, y)) for all fi in X
        rff = (1/(k*(k-1)))*sum(su(fi, fj)) for all fi and fj in X

        :param X:  {numpy array}, shape (n_samples, n_features) input data
        :param y:  {numpy array}, shape (n_samples) input class labels
        :return merits: {float}  merit of a feature subset X
        """

        n_samples, n_features = X.shape
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
        merits = rcf / np.sqrt(n_features + rff)
        return merits

    def information_gain(self, f1, f2):
        """
        This function calculates the information gain, where ig(f1, f2) = H(f1) - H(f1\f2)

        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: ig: {float}
        """

        ig = self.entropyd(f1) - self.conditional_entropy(f1, f2)
        return ig

    def conditional_entropy(self, f1, f2):
        """
        This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2)
        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: ce {float} conditional entropy of f1 and f2
        """

        ce = self.entropyd(f1) - self.midd(f1, f2)
        return ce

    def su_calculation(self, f1, f2):
        """
        This function calculates the symmetrical uncertainty, where su(f1,f2) = 2*IG(f1,f2)/(H(f1)+H(f2))
        :param f1: {numpy array}, shape (n_samples,)
        :param f2: {numpy array}, shape (n_samples,)
        :return: su {float} su is the symmetrical uncertainty of f1 and f2
        """
        # calculate information gain of f1 and f2, t1 = ig(f1, f2)
        t1 = self.information_gain(f1, f2)
        # calculate entropy of f1
        t2 = self.entropyd(f1)
        # calculate entropy of f2
        t3 = self.entropyd(f2)

        su = 2.0 * t1 / (t2 + t3)

        return su

    def entropyd(self, sx, base=2):
        """
        Discrete entropy estimator given a list of samples which can be any hashable object
        """

        return self.entropyfromprobs(self.hist(sx), base=base)

    @staticmethod
    def hist(sx):
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return map(lambda z: float(z) / len(sx), d.values())

    def entropyfromprobs(self, probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(self.elog, probs)) / log(base)

    @staticmethod
    def elog(x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0. or x >= 1.:
            return 0
        else:
            return x * log(x)

    def midd(self, x, y):
        """
        Discrete mutual information estimator given a list of samples which can be any hashable object
        """

        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)
