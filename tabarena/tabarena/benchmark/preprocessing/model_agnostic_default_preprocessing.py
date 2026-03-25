from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.features.generators.astype import AsTypeFeatureGenerator
from autogluon.features.generators.auto_ml_pipeline import (
    AutoMLPipelineFeatureGenerator,
)
from autogluon.features.generators.category import CategoryFeatureGenerator
from autogluon.features.generators.drop_duplicates import DropDuplicatesFeatureGenerator
from autogluon.features.generators.fillna import FillNaFeatureGenerator

from tabarena.benchmark.preprocessing.date_feature_generators import (
    DateTimeFeatureGenerator,
)
from tabarena.benchmark.preprocessing.text_feature_generators import (
    SemanticTextFeatureGenerator,
    StatisticalTextFeatureGenerator,
)

if TYPE_CHECKING:
    import pandas as pd

# TODO: we likely need some kind of off-loading logic for text features
class TabArenaModelAgnosticPreprocessing(AutoMLPipelineFeatureGenerator):
    """TabArena Model Agnostic Preprocessing."""

    def __init__(
        self,
        *,
        enable_datetime_features: bool = False,
        enable_text_ngram_features: bool = False,
        enable_text_special_features: bool = True,
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

        post_and_pre_handling = dict(  # noqa: C408
            # Fix string handling by passing own versions of the default pre-generators
            pre_generators=[
                StringFixAsTypeFeatureGenerator(),
                FillNaFeatureGenerator(),
                DropDuplicatesFeatureGenerator(),
            ],
            pre_enforce_types=False,
            # TODO: change such that text cols are skipped for duplicate check.
            #   Otherwise, duplicate check akes too long for text-use case, and we
            #   do not expect duplicates.
            post_drop_duplicates=False,
        )

        super().__init__(
            enable_text_ngram_features=enable_text_ngram_features,
            enable_datetime_features=enable_datetime_features,
            custom_feature_generators=custom_feature_generators,
            enable_text_special_features=enable_text_special_features,
            **post_and_pre_handling,
            **kwargs,
        )

    def _get_category_feature_generator(self):
        return NoCatAsStringCategoryFeatureGenerator()


class NoCatAsStringCategoryFeatureGenerator(CategoryFeatureGenerator):
    """CategoryFeatureGenerator that does not treat each string column as a category."""

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        X, type_group_map_special = super()._fit_transform(X=X, **kwargs)

        text_as_category_features = type_group_map_special.pop("text_as_category", None)
        X = (
            X.drop(columns=text_as_category_features)
            if text_as_category_features
            else X
        )

        return X, type_group_map_special

    def _generate_category_map(self, X: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        from autogluon.common.features.types import S_TEXT

        # Remove text features from input to cat maker
        type_group_map_special = self.feature_metadata_in.type_group_map_special
        if S_TEXT in type_group_map_special:
            text_features = type_group_map_special[S_TEXT]
            X = X.drop(columns=text_features)
            self._remove_features_in(text_features)

        return super()._generate_category_map(X=X)


class StringFixAsTypeFeatureGenerator(AsTypeFeatureGenerator):
    """Custom AsTypeFeatureGenerator to fix string dtype handling.

    The default string detection from AutoGluon is hardcoded in a weird way. Thus, we
    overwrite it here before passing feature metadata to the rest of the pipeline.

    Currently, we overwrite it such that we believe the dtype of the input dataframe.
    """

    # TODO: maybe better cardinality threshold but we assume we only
    #  run on well-curated data for now
    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> (pd.DataFrame, dict):
        X, type_group_map_special = super()._fit_transform(X=X, **kwargs)

        found_text_cols = type_group_map_special.get("text", [])
        found_text_cols += list(X.dtypes[["string" in str(x) for x in X.dtypes]].index)
        found_text_cols = list(set(found_text_cols))
        if found_text_cols:
            type_group_map_special["text"] = found_text_cols
        return X, type_group_map_special
