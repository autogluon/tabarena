import logging
import time

import numpy as np
import pandas as pd
from scipy.stats import entropy

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class GainRatioFeatureSelector(AbstractFeatureSelector):
    """
    GainRatio Feature Selection.

    Reference: Quinlan, J. Ross. "Induction of decision trees." Machine learning 1.1 (1986): 81-106.
    Implementation Source: Algorithm in the paper implemented by Bastian Schäfer
    """

    name = "GainRatioFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        n_samples, n_features = X.shape
        F = np.zeros(n_features, dtype=float)

        # Parent entropy H(Y)
        e_parent = self._entropy_from_counts(y.value_counts(dropna=False))  # [web:19][web:27]

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
                e_child += p * self._entropy_from_counts(y_sub.value_counts(dropna=False))  # [web:19][web:27]

            info_gain = e_parent - e_child

            F[i] = info_gain / split_info if split_info > 0 else 0.0

        feature_scores = dict(zip(X.columns, F))
        return feature_scores

    @staticmethod
    def _entropy_from_counts(counts: pd.Series) -> float:
        """
        Shannon entropy (bits) from counts.
        scipy.stats.entropy accepts (possibly unnormalized) event counts.
        """
        return float(entropy(counts.to_numpy(), base=2))
