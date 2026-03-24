from __future__ import annotations

from autogluon.features.generators.auto_ml_pipeline import (
    AutoMLPipelineFeatureGenerator,
)

from tabarena.benchmark.preprocessing.date_feature_generators import (
    DateTimeFeatureGenerator,
)
from tabarena.benchmark.preprocessing.text_feature_generators import (
    SemanticTextFeatureGenerator,
    StatisticalTextFeatureGenerator,
)


class TabArenaModelAgnosticPreprocessing(AutoMLPipelineFeatureGenerator):
    """TabArena Model Agnostic Preprocessing."""

    def __init__(
        self,
        *,
        enable_text_ngram_features: bool = False,
        enable_datetime_features: bool = False,
        enable_sematic_text_features: bool = True,
        enable_statistical_text_features: bool = True,
        enable_new_datetime_features: bool = True,
        **kwargs,
    ):
        """Custom init of the AutoMLPipelineFeatureGenerator with our new changes."""
        custom_feature_generators = []
        if enable_sematic_text_features:
            custom_feature_generators.append(SemanticTextFeatureGenerator())
        if enable_statistical_text_features:
            custom_feature_generators.append(StatisticalTextFeatureGenerator())
        if enable_new_datetime_features:
            custom_feature_generators.append(DateTimeFeatureGenerator())
        if len(custom_feature_generators) == 0:
            custom_feature_generators = None

        super().__init__(
            enable_text_ngram_features=enable_text_ngram_features,
            enable_datetime_features=enable_datetime_features,
            custom_feature_generators=custom_feature_generators,
            **kwargs,
        )
