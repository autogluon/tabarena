from __future__ import annotations

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
from tabarena.benchmark.preprocessing.group_feature_generators import (
    GroupAggregationFeatureGenerator,
)
from tabarena.benchmark.preprocessing.text_feature_generators import (
    SemanticTextFeatureGenerator,
    StatisticalTextFeatureGenerator,
)
from tabarena.benchmark.task.user_task import GroupLabelTypes


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
        group_cols: str | list[str] | None = None,
        group_labels: GroupLabelTypes | None = None,
        group_time_on: str | None = None,
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

        # TODO(future):
        #   - refactor such that we automatically detect group labels type.
        #   - We add it as a post-generator mostly to allow for dropping the group col.
        #       In theory, we could filter it differently via some dtype setting.
        post_generators = []
        if group_cols is not None:
            assert group_labels is not None, "If group_cols is specified, group_labels must also be specified."
            assert group_labels in [
                GroupLabelTypes.PER_GROUP,
                GroupLabelTypes.PER_SAMPLE,
            ], "group_labels must be either PER_GROUP or PER_SAMPLE if group_cols is specified."
            post_generators.append(
                GroupAggregationFeatureGenerator(
                    group_col=group_cols,
                    generate_index_features=group_labels == GroupLabelTypes.PER_GROUP,
                    group_time_on=group_time_on,
                )
            )

        if len(custom_feature_generators) == 0:
            custom_feature_generators = None
        if len(post_generators) == 0:
            post_generators = None

        post_and_pre_handling = dict(
            # Fix string handling by passing own versions of the default pre-generators
            pre_generators=[
                StringFixAsTypeFeatureGenerator(),
                FillNaFeatureGenerator(),
                DropDuplicatesFeatureGenerator(),
            ],
            post_generators=post_generators,
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
        # Pass categorical columns through *without* encoding.
        # Cat handling is deferred to TabArenaModelSpecificPreprocessing.
        return IdentityFeatureGenerator(
            infer_features_in_args={
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

    We further adjust the original logic to better handle unseen categories or suddenly appearing
    nan values at test time.

    """

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None, **kwargs) -> pd.DataFrame:
        """Rename columns with '.' before AutoGluon stores feature metadata.

        AutoGluon's ``AbstractFeatureGenerator.fit_transform`` records ``features_in``
        from the *original* X before calling ``_fit_transform``.  We must therefore
        rename at the public API level so that the stored metadata matches what the
        parent's ``_fit_transform`` will see.
        """
        self._dot_rename_map_: dict[str, str] = {c: str(c).replace(".", "_") for c in X.columns if "." in str(c)}
        if self._dot_rename_map_:
            X = X.rename(columns=self._dot_rename_map_)
        return super().fit_transform(X, y=y, **kwargs)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply the same dot-renaming as fit before passing to parent transform."""
        if self._dot_rename_map_:
            X = X.rename(columns=self._dot_rename_map_)
        return super().transform(X)

    def _handle_nan_in_int_only_at_test_time(self, X: pd.DataFrame) -> pd.DataFrame:
        """Handle int features that contain null values at inference time but not at fit time.
        This logic is copied from the original AsTypeFeatureGenerator._transform.
        """
        null_count = X[self._int_features].isnull().any()
        # If int feature contains null during inference but not during fit.
        if null_count.any():
            # TODO: Consider imputing to mode? This is tricky because training data had no missing values.
            # TODO: Add unit test for this situation, to confirm it is handled properly.
            with_null = null_count[null_count]
            with_null_features = list(with_null.index)
            self._log(
                20,
                "WARNING: Int features without null values "
                "at train time contain null values at inference time! "
                "Imputing nulls to 0. To avoid this, pass the features as floats during fit!",
            )
            self._log(10, f"WARNING: Int features with nulls: {with_null_features}")
            X[with_null_features] = X[with_null_features].fillna(0)

        return X

    def _handle_dtype_mismatch_at_test_time(self, X: pd.DataFrame, bool_cols_with_extra_cats: set) -> pd.DataFrame:
        """Handle situation where dtypes of test data do not match those of training data.

        The logic is split between cat and non-cat features to avoid the issue where
        astype(CategoricalDtype(categories=[...])) silently maps unknown categories to NaN.
        By converting through object dtype first, we ensure that all values are preserved as valid categories,
        even if they were not seen during training.
        bool_cols_with_extra_cats are excluded from non_cat_type_map because they
        are still typed as int8 in _type_map_real_opt but have not been bool-encoded;
        trying to astype them to int8 would silently discard the extra category values.
        """
        # TODO: Confirm this works with sparse and other feature types!
        # FIXME: Address situation where test-time invalid type values cause crash:
        #  https://stackoverflow.com/questions/49256211/how-to-set-unexpected-data-type-to-na?noredirect=1&lq=1
        # For categorical columns, astype(CategoricalDtype(categories=[...])) silently
        # maps unknown categories to NaN.  Convert through object dtype instead so all
        # values are preserved as valid categories.

        cat_type_map = {
            col: dtype for col, dtype in self._type_map_real_opt.items() if isinstance(dtype, pd.CategoricalDtype)
        }
        non_cat_type_map = {
            col: dtype
            for col, dtype in self._type_map_real_opt.items()
            if not isinstance(dtype, pd.CategoricalDtype) and col not in bool_cols_with_extra_cats
        }
        if non_cat_type_map:
            try:
                X = X.astype(non_cat_type_map)
            except Exception as e:
                self._log_invalid_dtypes(X=X)
                raise e
        for col, dtype in cat_type_map.items():
            if col in X.columns:
                X[col] = X[col].astype(object).astype(pd.CategoricalDtype(ordered=dtype.ordered))
        return X

    def _handle_bool_cols_with_extra_cats_at_test_time(self, X: pd.DataFrame) -> tuple[pd.DataFrame, set]:
        """Handle situation where bool columns gain extra categories at test time.

        If a bool column gains more than the expected 2 unique non-null values at test time,
        we skip bool-encoding for that column and convert it to categorical at the end of the transform method.
        This is because encoding a 3rd value through the bool path (== true_val → 1, else → 0) silently maps unknown categories to 0 (false).
        By skipping bool-encoding and converting to categorical, we ensure that all values are preserved as valid categories,
        even if they were not seen during training.
        """
        bool_cols_with_extra_cats = {
            col for col in self._bool_features if col in X.columns and X[col].dropna().nunique() > 2
        }
        if bool_cols_with_extra_cats:
            saved_extra = {col: self._bool_features.pop(col) for col in bool_cols_with_extra_cats}
            self._set_bool_features_val()
        if self._bool_features:
            X = self._convert_to_bool(X)
        if bool_cols_with_extra_cats:
            self._bool_features.update(saved_extra)
            self._set_bool_features_val()

        return X, bool_cols_with_extra_cats

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Override the default handling for unseen values!"""
        # Identify bool columns that gained more than the expected 2 unique non-null values
        # at test time.  Encoding a 3rd value through the bool path (== true_val → 1, else → 0)
        # silently maps unknown categories to 0 (false).  We instead skip bool-encoding for
        # those columns and convert them to categorical at the end of this method.
        bool_cols_with_extra_cats: set[str] = set()
        if self._bool_features:
            X, bool_cols_with_extra_cats = self._handle_bool_cols_with_extra_cats_at_test_time(X)

        # This means we have unobserved nans/categories
        if self._type_map_real_opt != X.dtypes.to_dict():
            if self._int_features.size:
                X = self._handle_nan_in_int_only_at_test_time(X)

            if self._type_map_real_opt:
                X = self._handle_dtype_mismatch_at_test_time(X, bool_cols_with_extra_cats=bool_cols_with_extra_cats)

        # Convert bool columns that gained extra categories to categorical so that
        # all values (including novel ones) are preserved rather than silently mapped to 0.
        for col in bool_cols_with_extra_cats:
            if col in X.columns:
                X[col] = X[col].astype(object).astype(pd.CategoricalDtype(ordered=False))

        return X

    def _fit_transform(self, X: pd.DataFrame, **kwargs) -> tuple[pd.DataFrame, dict]:
        # X arrives here with '.' already replaced by '_' (done in fit_transform above).
        X, type_group_map_special = super()._fit_transform(X=X, **kwargs)

        found_text_cols = type_group_map_special.get("text", [])
        found_text_cols += list(X.dtypes[["string" in str(x) for x in X.dtypes]].index)
        found_text_cols = list(set(found_text_cols))
        if found_text_cols:
            type_group_map_special["text"] = found_text_cols
        return X, type_group_map_special
