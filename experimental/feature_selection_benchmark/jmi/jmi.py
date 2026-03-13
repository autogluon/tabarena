import logging
import time

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class JMIFeatureSelector(AbstractFeatureSelector):
    """
    JMI Feature Selection.

    Reference: Yang, Howard, and John Moody. "Data visualization and feature selection: New algorithms for nongaussian data." Advances in neural information processing systems 12 (1999).
    Implementation Inspiration: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/JMI.py#L4.
                           The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
                           - Add time constraint
                           - Adapt implementation, so that
    """

    name = "JMIFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        """
        Implements Joint Mutual Information (JMI) feature selection using
        the Kullback-Leibler divergence definition:
        I(X_1,...,X_k ; Y) = KL( p(x1,...,xk, y) || p(x1,...,xk) * p(y) )
        """

        X_np = self._discretize(X.to_numpy(), time_limit, start_time)
        y_np = y.to_numpy()

        n_samples, n_features = X_np.shape
        selected = []  # indices of selected features
        remaining = list(range(n_features))
        scores = np.zeros(n_features)
        for i in remaining:
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            scores[i] = self._joint_mi_kl(X_np[:, [i]], y_np, time_limit, start_time)

        best_first = int(np.argmax(scores))
        selected.append(best_first)
        remaining.remove(best_first)

        while len(selected) < self.max_features and remaining:
            best_score = -np.inf
            best_idx = None
            for i in remaining:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                candidate_cols = selected + [i]
                X_subset = X_np[:, candidate_cols]
                score = self._joint_mi_kl(X_subset, y_np, time_limit, start_time)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    scores[i] = score
            selected.append(best_idx)
            remaining.remove(best_idx)
        feature_scores = dict(zip(X.columns, scores))
        return feature_scores

    @staticmethod
    def _discretize(X_np: np.ndarray, time_limit, start_time, n_bins: int = 10) -> np.ndarray:
        """Bin continuous features into integers for probability estimation."""
        X_disc = np.zeros_like(X_np, dtype=int)
        for i in range(X_np.shape[1]):
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            col = X_np[:, i]
            bins = np.linspace(col.min(), col.max(), n_bins)
            X_disc[:, i] = np.digitize(col, bins)
        return X_disc

    @staticmethod
    def _estimate_prob(data: np.ndarray, time_limit, start_time) -> dict:
        """Estimate joint probability distribution from rows of data."""
        if data.ndim == 1:
            data = data.reshape(-1, 1)
        n = data.shape[0]
        counts = {}
        for row in data:
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            key = tuple(row)
            counts[key] = counts.get(key, 0) + 1
        return {k: v / n for k, v in counts.items()}

    def _joint_mi_kl(self, X_subset: np.ndarray, y_np: np.ndarray, time_limit, start_time) -> float:
        """
        I(X_1,...,X_k ; Y) = KL( p(x,y) || p(x)*p(y) )
                           = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) )
        """
        if X_subset.ndim == 1:
            X_subset = X_subset.reshape(-1, 1)

        y_col = y_np.reshape(-1, 1)

        p_xy = self._estimate_prob(np.hstack([X_subset, y_col]), time_limit, start_time)
        p_x = self._estimate_prob(X_subset, time_limit, start_time)
        p_y = self._estimate_prob(y_col, time_limit, start_time)

        jmi = 0.0
        for xy_key, p_xy_val in p_xy.items():
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            x_key = xy_key[:-1]
            y_key = (xy_key[-1],)

            p_x_val = p_x.get(x_key, 0)
            p_y_val = p_y.get(y_key, 0)

            if p_x_val > 0 and p_y_val > 0:
                jmi += p_xy_val * np.log(p_xy_val / (p_x_val * p_y_val))

        return jmi
