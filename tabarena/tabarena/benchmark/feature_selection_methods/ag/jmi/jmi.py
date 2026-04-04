"""Joint Mutual Information (JMI) feature selection."""
from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class JMIFeatureSelector(AbstractFeatureSelector):
    """JMI Feature Selection.

    Reference: Yang, Howard, and John Moody. "Data visualization and
    feature selection: New algorithms for nongaussian data." Advances
    in neural information processing systems 12 (1999).
    Implementation Inspiration:
    https://github.com/jundongl/scikit-feature/blob/
    48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/
    function/information_theoretical_based/JMI.py#L4.
    The author of the code is Li, Jundong, Associate Professor at the
    University of Virginia and main-author of
    'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
        - Add time constraint
        - Use pandas instead of numpy and avoid conversion
        - Adapt implementation, so that JMI is calculated following
          the algorithm in the paper directly
    """

    name = "JMIFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        """Implement Joint Mutual Information (JMI) feature selection.

        Uses the Kullback-Leibler divergence definition:
        I(X_1,...,X_k ; Y) = KL( p(x1,...,xk, y) || p(x1,...,xk) * p(y) )
        """
        start_time = time.monotonic()

        X = self._discretize(X, time_limit, start_time)
        n_features = len(X.columns)
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
            scores[i] = self._joint_mi_kl(X.iloc[:, [i]], y, time_limit, start_time)

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
                candidate_cols = [*selected, i]
                X_subset = X.iloc[:, candidate_cols]
                score = self._joint_mi_kl(X_subset, y, time_limit, start_time)
                if score > best_score:
                    best_score = score
                    best_idx = i
                    scores[i] = score
            selected.append(best_idx)
            remaining.remove(best_idx)
        return dict(zip(X.columns, scores))

    @staticmethod
    def _discretize(X: pd.DataFrame, time_limit, start_time, n_bins: int = 10) -> pd.DataFrame:
        """Bin continuous features into integers for probability estimation."""
        X_disc = pd.DataFrame(np.zeros(X.shape, dtype="object"), index=X.index, columns=X.columns)
        numerical_cols = X.select_dtypes(include=["number"]).columns.tolist()
        for col_name in numerical_cols:
            i = X.columns.get_loc(col_name)
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            col_data = X.iloc[:, i]
            bins = np.linspace(col_data.min(), col_data.max(), n_bins)
            X_disc.iloc[:, i] = pd.cut(col_data, bins=bins, labels=False, right=False)
        return X_disc

    @staticmethod
    def _estimate_prob(data: pd.DataFrame, time_limit, start_time) -> dict:
        """Estimate joint probability distribution from rows of data."""
        n = len(data.columns)
        counts = {}
        for row in data:
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break
            counts[row] = counts.get(row, 0) + 1
        return {k: v / n for k, v in counts.items()}

    def _joint_mi_kl(self, X_subset: pd.DataFrame, y: pd.Series, time_limit, start_time) -> float:
        """I(X_1,...,X_k ; Y) = KL( p(x,y) || p(x)*p(y) )
        = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) ).
        """
        if X_subset.ndim == 1:
            X_subset = X_subset.reshape(-1, 1)

        X_subset_y = pd.concat([X_subset, y], axis=1)
        p_xy = self._estimate_prob(X_subset_y, time_limit, start_time)
        p_x = self._estimate_prob(X_subset, time_limit, start_time)
        p_y = self._estimate_prob(y.to_frame(), time_limit, start_time)

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
