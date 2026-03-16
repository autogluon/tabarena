import logging
import time
from math import log

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class MIFeatureSelector(AbstractFeatureSelector):
    """
    MI Feature Selection.

    Reference: Battiti, Roberto. "Using mutual information for selecting features in supervised neural net learning." IEEE Transactions on neural networks 5.4 (1994): 537-550.
    Implementation Inspiration: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/utility/mutual_information.py#L4
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Code adapted so that the formula in the paper is calculated directly, the parts of the formula are calculated using the entropy code of the implementation inspiration
    """

    name = "MIFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        X_np = X.to_numpy()

        n_samples, n_features = X_np.shape
        MI = np.zeros(n_features)

        t1 = np.zeros(n_features)
        t2 = np.zeros(n_features)
        t3 = np.zeros(n_features)
        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            f = X_np[:, i]
            t1[i] = self.entropyd(f)
            t2[i] = self.entropyd(y)
            t3[i] = self.midd(f, y)
            MI[i] = t1[i] + t2[i] - t3[i]

        feature_scores = dict(zip(X.columns, MI))
        return feature_scores

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