import logging
import time

import numpy as np
import pandas as pd

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)


class SequentialBackwardEliminationFeatureSelector(AbstractFeatureSelector):
    """
    SequentialBackwardElimination Feature Selection.

    Implementation Source: Algorithm implemented by Bastian Schäfer (including time constraint using the autogluon model)
    """

    name = "SequentialBackwardEliminationFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None) -> dict[str, float]:
        start_time = time.monotonic()
        current_features = self._original_features.copy()
        if self.max_features < len(self._original_features):
            while len(current_features) > self.max_features:
                elapsed_time = time.time() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break
                worst_score = -np.inf
                worst_feature = None

                for feature in current_features:
                    elapsed_time = time.time() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break
                    time_to_fit = None
                    if time_limit is not None:
                        time_to_fit = int(time_limit - time.monotonic() - start_time)
                    test_features = [f for f in current_features if f != feature]
                    test_X = X[test_features]
                    score = self.evaluate_proxy_model(X=test_X, y=y, time_limit=time_to_fit)
                    if score > worst_score:
                        worst_score = score
                        worst_feature = feature
                current_features.remove(worst_feature)
        else:
            current_features = self._original_features.copy()

        return [str(feat) for feat in current_features]
