import logging
import time

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class ImpurityFeatureSelector(AbstractFeatureSelector):
    """
    Impurity Feature Selection.

    Reference: (this is not the original source, even if it is cited a lot) Duch, W. (2006). Filter Methods. In: Guyon, I., Nikravesh, M., Gunn, S., Zadeh, L.A. (eds) Feature Extraction. Studies in Fuzziness and Soft Computing, vol 207. Springer, Berlin, Heidelberg. https://doi.org/10.1007/978-3-540-35488-8_4.
    Implementation Source: https://github.com/KhaosResearch/CMF-AGAwER/blob/d24e61e78ac197ad75342e8f4be5d63d17bd9e7a/CMF-AGAwER.py#L59
                           The author of the code is Hossein Nematzadeh, Associate Professor at the University of Malaga and main-author of 'A review of feature selection methods based on meta-heuristic algorithms' (2025).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Use pandas instead of numpy and avoid conversion
    """

    name = "ImpurityFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_features = len(X.columns)
        alpha = np.zeros(n_features)
        for w in range(n_features):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            sorted_indices = np.argsort(X.iloc[:, w])
            sorted_labels = y.iloc[sorted_indices]
            alpha[w] = self.classification_error_impurity(sorted_labels, time_limit, start_time)
        feature_scores = dict(zip(X.columns, alpha))
        return feature_scores

    @staticmethod
    def classification_error_impurity(arr, time_limit, start_time):
        unique_elements = np.unique(arr)
        w_values = []
        for current_element in unique_elements:
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            if current_element not in arr:
                continue
            indices = np.where(arr == current_element)[0]
            arr_element = arr[indices[0]:indices[-1] + 1]
            n_other = np.count_nonzero(arr_element != current_element)
            n_current = np.count_nonzero(arr_element == current_element)
            if n_current == 0:
                w_values.append(float('inf'))
            else:
                w = n_other / (n_current + n_other)
                w_values.append(w)
        return 1 - min(w_values)
