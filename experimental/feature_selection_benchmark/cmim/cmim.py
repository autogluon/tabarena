import logging
import time
from math import log

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class CMIMFeatureSelector(AbstractFeatureSelector):
    """
    CMIM Feature Selection.

    Reference: Fleuret, François. "Fast binary feature selection with conditional mutual information." Journal of Machine learning research 5.Nov (2004): 1531-1555.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/CMIM.py#L4.
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    """

    name = "CMIMFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X_np = X.to_numpy()
        y_np = y.to_numpy()

        n_samples, n_features = X.shape

        F = np.nan * np.zeros(n_features)
        CMIM = np.zeros(n_features)
        m = np.zeros(n_features) - 1

        # Init CMIM with Mutual Information
        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X_np[:, i]
            CMIM[i] = self.midd(f, y)

        for k in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            # Choose the feature with the highest MI as the next feature
            # if k == 0:
            idx = np.argmax(CMIM)
            F[k] = idx
            CMIM[idx] = -np.inf
            # Early stopping
            if np.sum(~np.isnan(F)) == self.max_features:
                break

            sstar = -1000000  # start with really low value for best partial score sstar
            for i in range(n_features):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                if i not in F:
                    while (CMIM[i] > sstar) and (m[i] < k - 1):
                        elapsed_time = time.time() - start_time
                        if (time_limit is not None) and (elapsed_time >= time_limit):
                            logger.warning(
                                f"Warning: FeatureSelection Method has no time left to train... "
                                f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                            )
                            break
                        m[i] = m[i] + 1
                        CMIM[i] = min(CMIM[i], self.cmidd(X_np[:, i], y_np, X_np[:, int(F[int(m[i])])]))
                    if CMIM[i] > sstar:
                        sstar = CMIM[i]
                        F[k + 1] = i
        selected_indices = [int(idx) for idx in F if not np.isnan(idx)]
        selected_features = [self._original_features[i] for i in selected_indices]
        return [str(feat) for feat in selected_features]

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
        return self.entropyd(list(zip(y, z))) + self.entropyd(list(zip(x, z))) - self.entropyd(
            list(zip(x, y, z))) - self.entropyd(z)

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