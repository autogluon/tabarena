from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


class SequentialForwardSelectionFeatureSelector(AbstractFeatureSelector):
    """SequentialForwardSelection Feature Selection.

    Implementation Source: Algorithm implemented by Bastian Schäfer (including time constraint using the autogluon model)
    """

    name = "SequentialForwardSelectionFeatureSelector"
    feature_scoring_method: bool = False

    def _fit_feature_selection(
        self, *, X: pd.DataFrame, y: pd.Series, time_limit: int | None = None
    ) -> dict[str, float]:
        start_time = time.monotonic()

        current_features = []
        if self.max_features < len(self._original_features):
            available_features = self._original_features.copy()
            while len(current_features) < self.max_features and available_features:
                elapsed_time = time.monotonic() - start_time
                if (time_limit is not None) and (elapsed_time >= time_limit):
                    logger.warning(
                        f"Warning: FeatureSelection Method has no time left to train... "
                        f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                    )
                    break

                best_score = -np.inf
                best_feature = None

                for feature in available_features:
                    elapsed_time = time.monotonic() - start_time
                    if (time_limit is not None) and (elapsed_time >= time_limit):
                        logger.warning(
                            f"Warning: FeatureSelection Method has no time left to train... "
                            f"\t(Time Elapsed = {elapsed_time:.1f}s, Time Limit = {time_limit:.1f}s)"
                        )
                        break

                    time_to_fit = None
                    if time_limit is not None:
                        time_to_fit = int(time_limit - elapsed_time * 1.1)  # buffer

                    test_features = [*current_features, feature]
                    test_X = X[test_features]
                    score = self.evaluate_proxy_model(X=test_X, y=y, time_limit=time_to_fit)

                    if score > best_score:
                        best_score = score
                        best_feature = feature

                if best_feature is None:
                    break

                current_features.append(best_feature)
                available_features.remove(best_feature)
        else:
            current_features = self._original_features.copy()
        return [str(feat) for feat in current_features]
