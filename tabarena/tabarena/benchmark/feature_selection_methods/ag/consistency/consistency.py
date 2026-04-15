"""Consistency-based feature selection."""
from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np
from tqdm import tqdm

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class ConsistencyFeatureSelector(AbstractFeatureSelector):
    """(In-)Consistency Feature Selection.

    Reference: Liu, H., & Setiono, R. (1996, July). A probabilistic approach to feature selection-a filter solution.
    In ICML (Vol. 96, pp. 319-327).
    Implementation Source: Algorithm in the paper implemented by Bastian Schäfer
    Changes to the algorithm by Bastian Schäfer:
                           - Add time constraint
                           - Add max_features (number of features to be maximally selected by the method) constraint
    """

    name = "ConsistencyFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> list[str]:
        start_time = time.monotonic()
        r = 77
        theta = 0.0
        _n_samples, n_features = X.shape
        rng = np.random.default_rng(1)

        c_best = n_features
        s_best = np.ones(n_features, dtype=bool)

        for _ in tqdm(range(r)):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break

            S = rng.random(n_features) < 0.5
            if not S.any():
                S[rng.integers(0, n_features)] = True

            # Enforce an upper bound on subset size
            if self.max_features is not None and S.sum() > self.max_features:
                on = np.where(S)[0]
                keep = rng.choice(on, size=self.max_features, replace=False)
                S[:] = False
                S[keep] = True
            C = int(S.sum())
            if c_best < C:
                continue

            IR = self._inconsistency_rate(X.loc[:, X.columns[S]], y, time_limit, start_time)
            if IR is None:
                selected_indices = np.where(s_best)[0].tolist() 
                selected_features = [self._original_features[i] for i in selected_indices]
                return [str(feat) for feat in selected_features] 
            if IR <= theta and c_best >= C:
                c_best = C
                s_best = S.copy()

        selected_indices = np.where(s_best)[0].tolist()
        selected_features = [self._original_features[i] for i in selected_indices]
        return [str(feat) for feat in selected_features]

    @staticmethod
    def _inconsistency_rate(X_sub: pd.DataFrame, y: pd.Series, time_limit, start_time) -> float:
        """IR(S) = (sum over patterns) (|pattern_group| - max_class_count) / n."""
        if X_sub.shape[1] == 0:
            return 1.0  # empty set cannot discriminate at all

        df = X_sub.copy()
        df["_y_"] = y.to_numpy()

        incons = 0
        for _, grp in df.groupby(list(X_sub.columns), dropna=False, observed=False):
            elapsed_time = time.monotonic() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                return None 
            counts = grp["_y_"].value_counts(dropna=False)
            incons += len(grp) - int(counts.max())

        return incons / len(df)
