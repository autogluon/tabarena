from __future__ import annotations

from copy import deepcopy

import pandas as pd
from autogluon.common.features.types import (
    R_BOOL,
    R_CATEGORY,
    R_OBJECT,
    S_DATETIME_AS_OBJECT,
    S_IMAGE_BYTEARRAY,
    S_IMAGE_PATH,
    S_TEXT,
    S_TEXT_SPECIAL,
)
from autogluon.features import BulkFeatureGenerator, IdentityFeatureGenerator
from autogluon.features.generators.category import CategoryFeatureGenerator

from tabarena.benchmark.preprocessing.text_feature_generators import (
    TextEmbeddingDimensionalityReductionFeatureGenerator,
)


class TabArenaModelSpecificPreprocessing:
    """Model-specific preprocessing.

    Applies two transformations on top of the model-agnostic feature generator:

    1. **Ordinal encoding** — ``CategoryFeatureGenerator`` maps ``R_CATEGORY``
       features to integer codes.  This step is deliberately absent from
       ``TabArenaModelAgnosticPreprocessing`` so that model classes that natively
       support categoricals (e.g. LightGBM, CatBoost) can decide how to consume
       them; model classes that require integers receive them here.

    2. **Text-embedding dimensionality reduction** —
       ``TextEmbeddingDimensionalityReductionFeatureGenerator`` compresses
       ``S_TEXT_EMBEDDING`` / ``S_TEXT_SPECIAL`` features with per-source-column
       PCA, keeping components that explain 99 % of variance (≤ 30 per group).
       All other features are passed through unchanged by ``IdentityFeatureGenerator``.
    """

    hp_key_kwargs: str = "ag.model_specific_feature_generator_kwargs"

    @staticmethod
    def add_to_hyperparameters(hyperparameters: dict) -> dict:
        """Inject the model-specific preprocessing into a *single model's* parameter dict.

        ``hyperparameters`` is the parameter dict for **one model** (e.g. ``{}`` for
        LightGBM with default settings).

        The returned dict gains the ``ag.model_specific_feature_generator_kwargs``
        key recognised by AutoGluon's ``AbstractModel``.  Callers that use
        ``TabularPredictor`` directly should therefore wrap the result themselves::

            gbm_params = TabArenaModelSpecificPreprocessing.add_to_hyperparameters({})
            predictor.fit(train_data, hyperparameters={"GBM": gbm_params})

        Inside the TabArena experiment framework (``AGSingleWrapper``) this wrapping
        happens automatically.
        """
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
        """Return a BulkFeatureGenerator that ordinal-encodes and reduces text embeddings.

        The three parallel generators inside the bulk group cover the full feature space:

        * ``CategoryFeatureGenerator`` — ordinal-encodes ``R_CATEGORY`` features.
        * ``IdentityFeatureGenerator`` — passes every other feature through (numerics,
          bools, raw text, datetime-derived), excluding text embeddings/specials handled
          by the DR generator below.
        * ``TextEmbeddingDimensionalityReductionFeatureGenerator`` — PCA-reduces
          ``S_TEXT_EMBEDDING`` / ``S_TEXT_SPECIAL`` features, grouped by source column.
        """
        # TODO: figure out how to more easily pass IdentityFeatureGenerator / dont drop other columns.
        bulk_kwargs = dict(
            generators=[
                # Cat/Ordinal Encoding
                [
                    # The other features are consumed, and thus can be dropped.
                    IdentityFeatureGenerator(
                        infer_features_in_args=NoCatAsStringCategoryFeatureGenerator.get_infer_features_in_args_to_drop()
                    ),
                    NoCatAsStringCategoryFeatureGenerator(),
                ],
                # PCA
                [
                    IdentityFeatureGenerator(
                        infer_features_in_args=TextEmbeddingDimensionalityReductionFeatureGenerator.get_infer_features_in_args_to_drop()
                    ),
                    TextEmbeddingDimensionalityReductionFeatureGenerator(),
                ],
            ],
            verbosity=2,
        )

        return [(BulkFeatureGenerator, bulk_kwargs)]


class NoCatAsStringCategoryFeatureGenerator(CategoryFeatureGenerator):
    """CategoryFeatureGenerator that does not treat each string column as a category.


    CategoryFeatureGenerator that preserves unseen categories to be handled by
    the downstream model instead of setting them to NaN.
    """

    def __init__(self, **kwargs) -> None:
        # Disable memory minimization to keep the original cat dtypes and pass them to the model
        # This avoids issues where we have to count up for unseen categories and leave it to the model to handle.
        kwargs.pop("minimize_memory", None)
        super().__init__(minimize_memory=False, **kwargs)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        X, type_group_map_special = super()._fit_transform(X=X, **kwargs)

        text_as_category_features = type_group_map_special.pop("text_as_category", None)
        X = X.drop(columns=text_as_category_features) if text_as_category_features else X

        return X, type_group_map_special

    def _generate_category_map(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        from autogluon.common.features.types import S_TEXT

        # Remove text features from input to cat maker
        # (Only a sanity check, should be removed from input)
        type_group_map_special = self.feature_metadata_in.type_group_map_special
        if S_TEXT in type_group_map_special:
            text_features = type_group_map_special[S_TEXT]
            X = X.drop(columns=text_features)
            self._remove_features_in(text_features)

        return super()._generate_category_map(X=X)

    def _generate_features_category(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.features_in:
            X_category = dict()
            if self.category_map is not None:
                for column, column_map in self.category_map.items():
                    col_values = X[column]
                    known = set(column_map)
                    # Detect non-NaN values absent from the train-time category set.
                    is_unseen = col_values.notna() & ~col_values.astype(object).isin(known)
                    if is_unseen.any():
                        # Keep the original unseen values by extending the category list with them.
                        col_values = col_values.astype(object)
                        unseen_vals = col_values[is_unseen].unique().tolist()
                        cats: pd.Index = pd.Index(list(column_map) + unseen_vals, dtype=object)
                    else:
                        cats = column_map
                    X_category[column] = pd.Categorical(col_values, categories=cats)
                X_category = pd.DataFrame(X_category, index=X.index)
                if self._fillna_map is not None:
                    for column, col_fill in self._fillna_map.items():
                        X_category[column] = X_category[column].fillna(col_fill)
        else:
            X_category = pd.DataFrame(index=X.index)
        return X_category

    @staticmethod
    def get_default_infer_features_in_args() -> dict:
        return {
            "valid_raw_types": [R_OBJECT, R_CATEGORY, R_BOOL],
            # Filter more than normally, as we also have text preprocessing
            # and we don't want to encode text-object columns.
            "invalid_special_types": [
                S_DATETIME_AS_OBJECT,
                S_IMAGE_PATH,
                S_IMAGE_BYTEARRAY,
                S_TEXT,
                S_TEXT_SPECIAL,
            ],
        }

    @staticmethod
    def get_infer_features_in_args_to_drop() -> dict:
        return {"invalid_raw_types": [R_OBJECT, R_CATEGORY, R_BOOL]}
