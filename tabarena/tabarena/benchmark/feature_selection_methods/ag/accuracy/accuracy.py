"""Accuracy-based feature selection."""
from __future__ import annotations

import time
from typing import TYPE_CHECKING

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd


class AccuracyFeatureSelector(AbstractFeatureSelector):
    """Accuracy-based Feature Selection."""

    name = "AccuracyFeatureSelector"
    feature_scoring_method: bool = True
    
    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        feature_scores = {}  # Store feature scores, higher is better
        
        time_to_fit = max(0.0, time_limit - (time.monotonic() - start_time)) if time_limit is not None else None
        baseline_score = self.evaluate_proxy_model(X=X, y=y, time_limit=time_to_fit)

        for feature in self._original_features:
            if self._timed_out(time_limit, start_time):
                break
            
            evaluate_X = X.drop(columns=[feature]).copy()    # Evaluate proxy model without the feature
            time_to_fit = max(0.0, time_limit - (time.monotonic() - start_time)) if time_limit is not None else None
            score = self.evaluate_proxy_model(X=evaluate_X, y=y, time_limit=time_to_fit)
            del evaluate_X  # free up memory

            # how much accuracy is lost without the feature (the higher the difference, the more important the feature)
            feature_scores[feature] = baseline_score - score

        return feature_scores