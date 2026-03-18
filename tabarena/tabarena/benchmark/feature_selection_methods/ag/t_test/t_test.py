from __future__ import annotations

import logging
import sys
import time
from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class tTestFeatureSelector(AbstractFeatureSelector):
    """t-Test Feature Selection.

    Reference: Peck R, Devore JL. Statistics: the exploration & analysis of data. Cengage learning. 2011; pp.516–9.
    Implementation Source: https://github.com/jundongl/scikit-feature/blob/master/skfeature/function/statistical_based/t_score.py
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "tTestFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        columns = X.columns
        n_features = len(X.columns)
        F = np.zeros(n_features)
        c = sorted(y.unique())
        if len(c) == 2:
            for i in range(n_features):
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                f = X.iloc[:, i]
                # class0 contains instances belonging to the first class
                # class1 contains instances belonging to the second class
                class0 = f[y == c[0]]
                class1 = f[y == c[1]]
                mean0 = class0.value_counts().max() / len(class0)  # Dominant class proportion
                mean1 = class1.value_counts().max() / len(class1)
                std0 = class0.nunique() / len(class0)  # "Dispersion"
                std1 = class1.nunique() / len(class1)
                n0 = len(class0)
                n1 = len(class1)
                t = mean0 - mean1
                t0 = (std0**2) / n0
                t1 = (std1**2) / n1
                F[i] = t / np.sqrt(t0 + t1)
        else:
            print("y should be guaranteed to a binary class vector")
            sys.exit(0)
        return dict(zip(columns, F))
