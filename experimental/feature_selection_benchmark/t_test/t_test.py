import logging
import time

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class tTestFeatureSelector(AbstractFeatureSelector):
    """
    t-Test Feature Selection.

    Reference: Peck R, Devore JL. Statistics: the exploration & analysis of data. Cengage learning. 2011; pp.516–9.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/statistical_based/t_score.py
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
    """

    name = "tTestFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        columns = X.columns
        X = X.to_numpy()
        n_samples, n_features = X.shape
        F = np.zeros(n_features)
        c = np.unique(y)
        if len(c) == 2:
            for i in range(n_features):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                f = X[:, i]
                # class0 contains instances belonging to the first class
                # class1 contains instances belonging to the second class
                class0 = f[y == c[0]]
                class1 = f[y == c[1]]
                mean0 = np.mean(class0)
                mean1 = np.mean(class1)
                std0 = np.std(class0)
                std1 = np.std(class1)
                n0 = len(class0)
                n1 = len(class1)
                t = mean0 - mean1
                t0 = np.true_divide(std0 ** 2, n0)
                t1 = np.true_divide(std1 ** 2, n1)
                F[i] = np.true_divide(t, (t0 + t1) ** 0.5)
        else:
            print('y should be guaranteed to a binary class vector')
            exit(0)
        feature_scores = dict(zip(columns, F))
        return feature_scores
