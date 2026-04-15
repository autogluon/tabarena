"""Conditional Mutual Information Maximization (CMIM) feature selection."""
from __future__ import annotations

import logging
import time
from math import log
from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class CMIMFeatureSelector(AbstractFeatureSelector):
    """CMIM Feature Selection.

    Reference: Fleuret, François. "Fast binary feature selection with
    conditional mutual information." Journal of Machine learning
    research 5.Nov (2004): 1531-1555.
    Implementation Source:
    https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/CMIM.py#L4.
    The author of the code is Li, Jundong, Associate Professor at the
    University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Add max_features (number of features to be maximally selected by the method) constraint
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "CMIMFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(  # noqa: C901
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> dict[str, float]:
        start_time = time.monotonic()
        n_features = len(X.columns)
        F = np.nan * np.zeros(n_features)
        CMIM = np.zeros(n_features)
        m = np.zeros(n_features) - 1
        F_set = set() 
        
        cols = [X.iloc[:, i] for i in range(n_features)]
        
        def timed_out() -> bool:  # helper for time-limit checks
            return time_limit is not None and (time.monotonic() - start_time) >= time_limit

        # Init CMIM with Mutual Information
        for i in range(n_features):
            if timed_out():
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {time.monotonic() - start_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            CMIM[i] = self.midd(cols[i], y)

        for k in range(n_features):
            if timed_out():
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {time.monotonic() - start_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            # Choose the feature with the highest MI as the next feature
            idx = np.argmax(CMIM)
            F[k] = idx
            F_set.add(int(idx))
            CMIM[idx] = -np.inf
            # Early stopping
            if k + 1 == self.max_features:
                break

            sstar = -np.inf # start with really low value for best partial score sstar
            for i in range(n_features):
                if timed_out():
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {time.monotonic() - start_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                if i not in F_set:
                    while (CMIM[i] > sstar) and (m[i] < k - 1):
                        if timed_out():
                            logger.warning(
                                f"Warning: FeatureSelection Method has no time left to train... "
                                f"\t(Time Elapsed = {time.monotonic() - start_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                            )
                            break
                        m[i] = m[i] + 1
                        CMIM[i] = min(CMIM[i], self.cmidd(cols[i], y, cols[int(F[int(m[i])])]))
                    if CMIM[i] > sstar:
                        sstar = CMIM[i]
        selected_indices = [int(idx) for idx in F if not np.isnan(idx)]
        selected_features = [self._original_features[i] for i in selected_indices]
        return [str(feat) for feat in selected_features]

    def conditional_entropy(self, f1, f2):
        """This function calculates the conditional entropy, where ce = H(f1) - I(f1;f2).

        Input
        -----
        f1: {numpy array}, shape (n_samples,)
        f2: {numpy array}, shape (n_samples,)

        Output
        ------
        ce: {float}
            ce is conditional entropy of f1 and f2
        """
        return self.entropyd(f1) - self.midd(f1, f2)

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

    @staticmethod
    def hist(sx):
        """Compute histogram (probability distribution) from a list of samples."""
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return (float(z) / len(sx) for z in d.values())

    def entropyfromprobs(self, probs, base=2):
        """Compute entropy from a probability distribution."""
        return -sum(map(self.elog, probs)) / log(base)

    @staticmethod
    def elog(x):
        """Compute x*log(x), returning 0 for x <= 0 or x >= 1."""
        if x <= 0.0 or x >= 1.0:
            return 0
        return x * log(x)
