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


class DISRFeatureSelector(AbstractFeatureSelector):
    """DISR Feature Selection.

    Reference: Meyer, Patrick E., and Gianluca Bontempi. "On the use of variable complementarity for feature selection in cancer classification." Workshops on applications of evolutionary computation. Berlin, Heidelberg: Springer Berlin Heidelberg, 2006.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/DISR.py#L5.
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Code adapted so that the formula in the paper is used, the parts of the formula are calculated using the code of the implementation source
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "DISRFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_features = len(X.columns)
        DISR = np.zeros(n_features)
        mutual_information = np.zeros(n_features)
        entropy = np.zeros(n_features)
        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X.iloc[:, i]
            mutual_information[i] = self.midd(f, y)
            entropy[i] = self.entropyd(list(zip(f, y)))
            symmetrical_relevance = mutual_information[i] / entropy[i]
            DISR[i] = symmetrical_relevance
        return dict(zip(X.columns, DISR))

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
        # Histogram from list of samples
        d = dict()
        for s in sx:
            d[s] = d.get(s, 0) + 1
        return (float(z) / len(sx) for z in d.values())

    def entropyfromprobs(self, probs, base=2):
        # Turn a normalized list of probabilities of discrete outcomes into entropy (base 2)
        return -sum(map(self.elog, probs)) / log(base)

    @staticmethod
    def elog(x):
        # for entropy, 0 log 0 = 0. but we get an error for putting log 0
        if x <= 0.0 or x >= 1.0:
            return 0
        return x * log(x)
