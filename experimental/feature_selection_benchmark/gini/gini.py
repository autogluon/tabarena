import logging
import time

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class GiniFeatureSelector(AbstractFeatureSelector):
    """
    Gini Index Feature Selection.

    Reference: Gini, Corrado W. "Variability and mutability, contribution to the study of statistical distributions and relations." Studi Economico-Giuridici della R. Universita de Cagliari (1912).
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/statistical_based/gini_index.py#L4.
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
    """

    name = "GiniFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        columns = X.columns
        X = X.to_numpy()
        y = y.to_numpy()
        n_samples, n_features = X.shape
        gini = np.ones(n_features) * 0.5
        for i in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(f"Warning: FeatureSelection Method has no time left to train... "
                               f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)")
                break
            v = np.unique(X[:, i])
            for j in range(len(v)):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(f"Warning: FeatureSelection Method has no time left to train... "
                                   f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)")
                    break
                # left_y contains labels of instances whose i-th feature value is less than or equal to v[j]
                left_y = y[X[:, i] <= v[j]]
                # right_y contains labels of instances whose i-th feature value is larger than v[j]
                right_y = y[X[:, i] > v[j]]

                # gini_left is sum of square of probability of occurrence of v[i] in left_y
                # gini_right is sum of square of probability of occurrence of v[i] in right_y
                gini_left = 0
                gini_right = 0

                for k in range(np.min(y), np.max(y) + 1):
                    elapsed_time = time.time() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(f"Warning: FeatureSelection Method has no time left to train... "
                                       f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)")
                        break
                    if len(left_y) != 0:
                        # t1_left is probability of occurrence of k in left_y
                        t1_left = np.true_divide(len(left_y[left_y == k]), len(left_y))
                        t2_left = np.power(t1_left, 2)
                        gini_left += t2_left
                    if len(right_y) != 0:
                        # t1_right is probability of occurrence of k in left_y
                        t1_right = np.true_divide(len(right_y[right_y == k]), len(right_y))
                        t2_right = np.power(t1_right, 2)
                        gini_right += t2_right

                    gini_left = 1 - gini_left
                    gini_right = 1 - gini_right

                    # weighted average of len(left_y) and len(right_y)
                    t1_gini = (len(left_y) * gini_left + len(right_y) * gini_right)

                    # compute the gini_index for the i-th feature
                    value = np.true_divide(t1_gini, len(y))

                    if value < gini[i]:
                        gini[i] = value
        feature_scores = dict(zip(columns, gini))
        return feature_scores
