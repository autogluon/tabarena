"""Gain ratio feature selection."""
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


class GainRatioFeatureSelector(AbstractFeatureSelector):
    """GainRatio Feature Selection.

    Reference: Quinlan, J. Ross. "Induction of decision trees."
    Machine learning 1.1 (1986): 81-106.
    Implementation Source: Algorithm in the paper implemented by
    Bastian Schäfer, using the entropy calculation of
    https://github.com/jundongl/scikit-feature/blob/
    48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/
    function/information_theoretical_based.
    The author of the entropy code is Li, Jundong, Associate Professor
    at the University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the algorithm by Bastian Schäfer:
                           - Add time constraint
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "GainRatioFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_features = len(X.columns)
        F = np.zeros(n_features, dtype=float)

        # Parent entropy H(Y)
        e_parent = self.entropyd(y.value_counts(dropna=False).values)

        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X.iloc[:, i]

            # SplitInfo(X_i) = - sum_v p(v) log2 p(v)
            p_v = f.value_counts(normalize=True, dropna=False)
            split_info = -(p_v * np.log2(p_v)).sum()

            # Conditional entropy H(Y | X_i) = sum_v p(v) H(Y | X_i=v)
            e_child = 0.0
            for v, p in p_v.items():
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                y_sub = y[f.eq(v)]
                e_child += p * self.entropyd(y_sub.value_counts(dropna=False).values)

            info_gain = e_parent - e_child

            F[i] = info_gain / split_info if split_info > 0 else 0.0

        return dict(zip(X.columns, F))

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

    def _entropy_from_counts(self, counts: pd.Series) -> float:
        """Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts.
        """
        return self.entropyd(counts.values)
