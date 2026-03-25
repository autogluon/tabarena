from __future__ import annotations

from copy import deepcopy

from autogluon.features import IdentityFeatureGenerator

from tabarena.benchmark.preprocessing.text_feature_generators import (
    TextEmbeddingDimensionalityReductionFeatureGenerator,
)


class TabArenaModelSpecificPreprocessing:
    """Model specific preprocessing (reduce dimensionality of text embeddings with PCA)."""

    hp_key_kwargs: str = "ag.model_specific_feature_generator_kwargs"

    @staticmethod
    def add_to_hyperparameters(hyperparameters: dict) -> dict:
        """Inject the model specific preprocessing into model hyperparameters."""
        hyperparameters = deepcopy(hyperparameters)
        hp_key_kwargs = TabArenaModelSpecificPreprocessing.hp_key_kwargs

        if hp_key_kwargs not in hyperparameters:
            hyperparameters[hp_key_kwargs] = {}
        if "feature_generators" not in hyperparameters[hp_key_kwargs]:
            hyperparameters[hp_key_kwargs]["feature_generators"] = []

        hyperparameters[hp_key_kwargs]["feature_generators"] += (
            TabArenaModelSpecificPreprocessing.get_model_specific_generator()
        )
        return hyperparameters

    @staticmethod
    def get_model_specific_generator() -> list:
        """Grouped PCA Generator with passthrough for non-text-embedding features."""
        filter_dtypes = TextEmbeddingDimensionalityReductionFeatureGenerator.get_infer_features_in_args_to_drop()[
            "invalid_special_types"
        ]
        return [
            # Passthrough for all non-text-embedding features
            (
                IdentityFeatureGenerator,
                dict(  # noqa: C408
                    infer_features_in_args={
                        "invalid_special_types": filter_dtypes,
                    }
                ),
            ),
            # PCA for text-embedding features
            (TextEmbeddingDimensionalityReductionFeatureGenerator, {}),
        ]
