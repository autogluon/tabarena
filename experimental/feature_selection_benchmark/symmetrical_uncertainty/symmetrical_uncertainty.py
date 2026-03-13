import logging
import time
from math import log

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class SymmetricalUncertaintyFeatureSelector(AbstractFeatureSelector):
    """
    Symmetrical Uncertainty Feature Selection.

    Reference: (this is not the original source, even if it is cited a lot) Flannery, Brian P., et al. "Numerical recipes in C." Press Syndicate of the University of Cambridge, New York 24.78 (1992): 36.
    Implementation Inspiration: Information Gain code from https://github.com/Thijsvanede/info_gain/blob/master/info_gain/info_gain.py & Entropy code from https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based.
                           The author of the Information Gain code is Thijs van Ede, Associate Professor at the University of Twente and main-author of 'FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted Network Traffic' (2020), where they used information gain
                           The author of the Entropy code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Adapt Information Gain and Entropy Code to Symmetrical Uncertainty algorithm presented in the paper
                           - Add time constraint
    """

    name = "SymmetricalUncertaintyFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        """
                Symmetrical Uncertainty for each feature:
                  SU(X, Y) = 2 * IG(Y|X) / (H(X) + H(Y))  (information gain / mutual information)
                where:
                  IG(Y|X) = H(Y) - H(Y|X)
        """
        start_time = time.monotonic()
        n_samples, n_features = X.shape
        SU = np.zeros(n_features, dtype=float)

        # H(Y)
        H_y = self.entropyd(y.value_counts(dropna=False).values)

        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X.iloc[:, i]
            # H(X)
            H_x = self.entropyd(f.value_counts(dropna=False).values)

            # H(Y|X) = sum_v p(v) * H(Y | X=v)
            p_v = f.value_counts(normalize=True, dropna=False)
            H_y_given_x = 0.0
            for v, p in p_v.items():
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break

                y_sub = y[f.eq(v)]
                H_y_given_x += p * self.entropyd(y_sub.value_counts(dropna=False).values)

            IG = H_y - H_y_given_x  # IG(Y|X) = H(Y) - H(Y|X)

            MI = H_x + H_y  # MI = (H(X) + H(Y)
            SU[i] = (2.0 * IG / MI) if MI > 0 else 0.0

        feature_scores = dict(zip(X.columns, SU))
        return feature_scores

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
