"""Minimum Redundancy Maximum Relevance (mRMR) feature selection."""
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


class mRMRFeatureSelector(AbstractFeatureSelector):  # noqa: N801
    """mRMR Feature Selection.

    Reference: Peng, Hanchuan, Fuhui Long, and Chris Ding. "Feature
    selection based on mutual information criteria of max-dependency,
    max-relevance, and min-redundancy." IEEE Transactions on pattern
    analysis and machine intelligence 27.8 (2005): 1226-1238.
    Implementation Inspiration:
    https://github.com/jundongl/scikit-feature/blob/
    48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/
    function/information_theoretical_based/MRMR.py
    The author of the code is Li, Jundong, Associate Professor at the
    University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
        - Add time constraint
        - Code adapted so that the formula in the paper is calculated
          directly, the parts of the formula are calculated using the
          entropy code of the implementation inspiration
        - Use pd.DataFrame directly instead of np.array
    """

    name = "mRMRFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_features = len(X.columns)
        relevance = np.zeros(n_features)
        redundancy = np.zeros(n_features)
        mRMR_score = np.zeros(n_features)
        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X.iloc[:, i]
            relevance[i] = self.midd(f, y)
            for j in range(n_features):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                if j != i:
                    f_j = X.iloc[:, j]
                    redundancy[i] += self.midd(f_j, f)
            mRMR_score[i] += relevance[i] - redundancy[i]

        return dict(zip(X.columns, -mRMR_score))

    def midd(self, x, y):
        """Discrete mutual information estimator given a list of samples which can be any hashable object."""
        return -self.entropyd(list(zip(x, y))) + self.entropyd(x) + self.entropyd(y)

    def entropyd(self, sx, base=2):
        """Discrete entropy estimator given a list of samples which can be any hashable object."""
        return self.entropyfromprobs(self.hist(sx), base=base)

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
