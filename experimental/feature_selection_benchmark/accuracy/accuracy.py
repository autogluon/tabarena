import logging
import time

import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)

class AccuracyFeatureSelector(AbstractFeatureSelector):
    """Accuracy-based Feature Selection."""

    name = "AccuracyFeatureSelector"
    feature_scoring_method: bool = True

    def _fit_feature_scoring(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()

        # Store feature scores, higher is better
        feature_scores = {}

        for feature in self._original_features:
            # Evaluate proxy model without the feature
            evaluate_X = X.drop(columns=[feature]).copy()

            time_to_fit = None
            if time_limit is not None:
                time_to_fit = int(time_limit - time.monotonic() - start_time)

            score = self.evaluate_proxy_model(X=evaluate_X, y=y, time_limit=time_to_fit)
            del evaluate_X  # free up memory

            # We want to keep the features that lead to the highest drop in score,
            # so we use the negative of the score.
            feature_scores[feature] = -score

            # Check time limit
            elapsed_time = time.time() - start_time
            if (time_limit is not None) and (elapsed_time >= time_limit):
                logger.warning(
                    f"Warning: FeatureSelection Method has no time left to train... "
                    f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                )
                break

        return feature_scores