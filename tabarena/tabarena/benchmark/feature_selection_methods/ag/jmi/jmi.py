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

    Reference: Yang, Howard, and John Moody. "Data visualization and feature selection: New algorithms for nongaussian data." 
    Advances in neural information processing systems 12 (1999).
    Implementation Inspiration: https://github.com/jundongl/scikit-feature/blob/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/function/information_theoretical_based/JMI.py#L4.
    The author of the code is Li, Jundong, Associate Professor at the University of Virginia and main-author of 'Feature selection: A data perspective' (2017).
    Changes to the implementation by Bastian Schäfer:
        - Add time constraint
        - Use pandas instead of numpy and avoid conversion
        - Adapt implementation, so that JMI is calculated following
          the algorithm in the paper directly
    """

    name = "JMIFeatureSelector"
    feature_scoring_method: bool = True

    @staticmethod
    def _timed_out(time_limit, start_time) -> bool:
        if time_limit is None:
            return False
        elapsed = time.monotonic() - start_time
        if elapsed >= time_limit:
            logger.warning(
                f"Warning: FeatureSelection Method has no time left to train... "
                f"\t(Time Elapsed = {elapsed:.1f}s, Time Limit = {time_limit:.1f}s)"
            )
            return True
        return False
    
    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        """Implement Joint Mutual Information (JMI) feature selection.
        
        Step 1: select first feature by plain MI: i1 = argmax_i I(Xi; Y)
        Step 2: select subsequent features by maximizing the sum of pairwise joint MI given selected features:
                i_k = argmax_i  sum_{j in selected} I(Xi, Xj; Y)
        """
        start_time = time.monotonic()

        X = self._discretize(X, time_limit, start_time)
        n_features = len(X.columns)
        selected = []  # indices of selected features
        remaining = list(range(n_features))
        scores = np.zeros(n_features)

        # step 1
        for i in remaining:
            if self._timed_out(time_limit, start_time): 
                break
            scores[i] = self._joint_mi_kl(X.iloc[:, [i]], y, time_limit, start_time)

        best_first = int(np.argmax(scores))
        selected.append(best_first)
        remaining.remove(best_first)

        # step 2
        while len(selected) < self.max_features and remaining:
            if self._timed_out(time_limit, start_time): 
                break

            best_score = -np.inf
            best_idx = None
            
            for i in remaining:
                if self._timed_out(time_limit, start_time): 
                    break

                score = sum(self._joint_mi_kl(X.iloc[:, [i, j]], y, time_limit, start_time) for j in selected)                
                if score > best_score:
                    best_score = score
                    best_idx = i
                    scores[i] = score
                
            if best_idx is None:
                break

            selected.append(best_idx)
            remaining.remove(best_idx)

        return dict(zip(X.columns, scores))


    @staticmethod
    def _discretize(X: pd.DataFrame, time_limit, start_time, n_bins: int = 10) -> pd.DataFrame:
        """Bin continuous features into integers for probability estimation."""
        X_disc = X.copy()
        for col_name in X.select_dtypes(include=["number"]).columns: 
            if JMIFeatureSelector._timed_out(time_limit, start_time): 
                break
            col_data = X[col_name] 
            bins = np.linspace(col_data.min(), col_data.max(), n_bins)
            X_disc[col_name] = pd.cut(col_data, bins=bins, labels=False, right=True, include_lowest=True)  
        return X_disc


    @staticmethod
    def _estimate_prob(data: pd.DataFrame) -> dict:
        """Estimate joint probability distribution from rows of data."""
        return data.value_counts(normalize=True).to_dict()


    def _joint_mi_kl(self, X_subset: pd.DataFrame, y: pd.Series, time_limit, start_time) -> float:
        """I(X_1,...,X_k ; Y) = KL( p(x,y) || p(x)*p(y) )
        = sum_{x,y} p(x,y) * log( p(x,y) / (p(x)*p(y)) ).
        """
        X_subset_y = pd.concat([X_subset, y], axis=1)
        p_xy = self._estimate_prob(X_subset_y)
        p_x = self._estimate_prob(X_subset)
        p_y = self._estimate_prob(y.to_frame())

        jmi = 0.0
        for xy_key, p_xy_val in p_xy.items():
            if JMIFeatureSelector._timed_out(time_limit, start_time): 
                break
            
            p_x_val = p_x.get(xy_key[:-1], 0)   # CHANGED: inlined x_key
            p_y_val = p_y.get((xy_key[-1],), 0)  
            if p_x_val > 0 and p_y_val > 0:
                jmi += p_xy_val * np.log(p_xy_val / (p_x_val * p_y_val))

        return jmi
