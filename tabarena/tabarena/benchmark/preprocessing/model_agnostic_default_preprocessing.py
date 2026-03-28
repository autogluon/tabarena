from __future__ import annotations

from typing import TYPE_CHECKING

from autogluon.common.features.types import R_CATEGORY
from autogluon.features import IdentityFeatureGenerator
from autogluon.features.generators.astype import AsTypeFeatureGenerator
from autogluon.features.generators.auto_ml_pipeline import (
    AutoMLPipelineFeatureGenerator,
)
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
            #   Otherwise, duplicate check take too long for text-use case, and we
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
        # Pass categorical columns through *without* ordinal encoding.
        # Ordinal encoding is deferred to TabArenaModelSpecificPreprocessing.
        return IdentityFeatureGenerator(
            infer_features_in_args={"valid_raw_types": [R_CATEGORY]}
        )


# TODO: maybe better cardinality threshold but we assume we only
#  run on well-curated data for now
class StringFixAsTypeFeatureGenerator(AsTypeFeatureGenerator):
    """Custom AsTypeFeatureGenerator to fix string dtype handling and column name sanitization.

    The default string detection from AutoGluon is hardcoded in a weird way. Thus, we
    overwrite it here before passing feature metadata to the rest of the pipeline.
    We overwrite it such that we believe the dtype of the input dataframe.

    Additionally, any input column whose name contains ``"."`` is renamed so that
    ``"."`` is replaced by ``"_"``.  The ``"."`` character is reserved as the
    source-column separator in text feature names produced downstream (e.g.
    ``TextSpecialFeatureGenerator`` produces ``{col}.char_count``).  Sanitizing
    raw column names here prevents parsing ambiguity in
    ``TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column``.
    """

    def fit_transform(
        self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs
    ) -> pd.DataFrame:
        """Rename columns with '.' before AutoGluon stores feature metadata.

        AutoGluon's ``AbstractFeatureGenerator.fit_transform`` records ``features_in``
        from the *original* X before calling ``_fit_transform``.  We must therefore
        rename at the public API level so that the stored metadata matches what the
        parent's ``_fit_transform`` will see.
        """
        self._dot_rename_map_: dict[str, str] = {
            c: str(c).replace(".", "_") for c in X.columns if "." in str(c)
        }
        if self._dot_rename_map_:
            X = X.rename(columns=self._dot_rename_map_)
        return super().fit_transform(X, y=y, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same dot-renaming as fit before passing to parent transform."""
        if self._dot_rename_map_:
            X = X.rename(columns=self._dot_rename_map_)
        return super().transform(X)

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> (pd.DataFrame, dict):
        # X arrives here with '.' already replaced by '_' (done in fit_transform above).
        X, type_group_map_special = super()._fit_transform(X=X, **kwargs)

        found_text_cols = type_group_map_special.get("text", [])
        found_text_cols += list(X.dtypes[["string" in str(x) for x in X.dtypes]].index)
        found_text_cols = list(set(found_text_cols))
        if found_text_cols:
            type_group_map_special["text"] = found_text_cols
        return X, type_group_map_special
