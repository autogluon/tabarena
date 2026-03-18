import logging

from experimental.feature_selection_benchmark.run_autogluon_feature_selection_pipeline import AbstractFeatureSelector

logger = logging.getLogger(__name__)

# TODO: ensure that we get a different random state each time we call this code within TabArena
class RandomFeatureSelector(AbstractFeatureSelector):
    """Random Feature Selection."""

    name = "RandomFeatureSelector"

    def _fit_feature_selection(self, **kwargs) -> list[str]:
        return self.fallback_feature_selection()