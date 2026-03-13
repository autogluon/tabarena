import logging
import time
from math import log

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class InformationGainFeatureSelector(AbstractFeatureSelector):
    """
    InformationGain Feature Selection.

    Reference: Quinlan, J. Induction of Decision Trees. Mach Learn 1, 81–106 (1986). https://doi.org/10.1023/A:1022643204877
    Implementation Inspritation: https://github.com/Thijsvanede/info_gain/blob/master/info_gain/info_gain.py & using entropy calculation of https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based.
                           The author of the information gain code is Thijs van Ede, Associate Professor at the University of Twente and main-author of 'FlowPrint: Semi-Supervised Mobile-App Fingerprinting on Encrypted Network Traffic' (2020), where they used information gain
                           The author of the entropy code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Add max_features (number of features to be maximally selected by the method) constraint
    """

    name = "InformationGainFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_samples, n_features = X.shape
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

            p_v = f.value_counts(normalize=True, dropna=False)
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

            F[i] = info_gain

        feature_scores = dict(zip(X.columns, F))
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

    def _entropy_from_counts(self, counts: pd.Series) -> float:
        """
        Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts.
        """
        return self.entropyd(counts.values)
