"""Random feature selection."""
from __future__ import annotations

import logging

from tabarena.benchmark.feature_selection_methods.abstract.abstract_feature_selector import AbstractFeatureSelector

logger = logging.getLogger(__name__)

# TODO: ensure that we get a different random state each time we call this code within TabArena
class RandomFeatureSelector(AbstractFeatureSelector):
    """Random Feature Selection."""

    name = "RandomFeatureSelector"

    def _fit_feature_selection(self, **kwargs) -> list[str]:  # noqa: ARG002
        return self.fallback_feature_selection()