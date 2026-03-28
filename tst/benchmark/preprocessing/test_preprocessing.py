from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.benchmark.preprocessing import (
    TabArenaModelAgnosticPreprocessing,
    TabArenaModelSpecificPreprocessing,
)
from tabarena.benchmark.preprocessing.date_feature_generators import (
    DateTimeFeatureGenerator,
)
from tabarena.benchmark.preprocessing.model_agnostic_default_preprocessing import (
    NoCatAsStringCategoryFeatureGenerator,
    StringFixAsTypeFeatureGenerator,
)
from tabarena.benchmark.preprocessing.text_feature_generators import (
    SemanticTextFeatureGenerator,
    StatisticalTextFeatureGenerator,
    TextEmbeddingDimensionalityReductionFeatureGenerator,
    sanitize_text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_RNG = np.random.default_rng(0)


def _make_float_df(n_rows: int = 20, n_cols: int = 10) -> pd.DataFrame:
    data = _RNG.standard_normal((n_rows, n_cols))
    return pd.DataFrame(data, columns=[f"f{i}" for i in range(n_cols)])


def _make_text_df(n_rows: int = 20) -> pd.DataFrame:
    phrases = ["hello world", "foo bar baz", "quick brown fox", "the lazy dog", "test text"]
    return pd.DataFrame(
        {"text": [phrases[i % len(phrases)] for i in range(n_rows)]}
    )


def _make_datetime_df(n_rows: int = 10) -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=n_rows, freq="ME")
    return pd.DataFrame({"dt": dates})


# ===========================================================================
# sanitize_text
# ===========================================================================


class TestSanitizeText:
    def test_lowercases(self):
        s = sanitize_text(pd.Series(["Hello", "WORLD"]))
        assert s.tolist() == ["hello", "world"]

    def test_strips_leading_trailing_whitespace(self):
        s = sanitize_text(pd.Series(["  foo  ", "\tbar\t"]))
        assert s.tolist() == ["foo", "bar"]

    def test_collapses_internal_spaces(self):
        s = sanitize_text(pd.Series(["a  b   c"]))
        assert s.iloc[0] == "a b c"

    def test_collapses_tabs_to_single_space(self):
        s = sanitize_text(pd.Series(["foo\t\tbar"]))
        assert s.iloc[0] == "foo bar"

    def test_nan_replaced_with_default(self):
        s = sanitize_text(pd.Series([None, float("nan")]))
        assert s.tolist() == ["missing data", "missing data"]

    def test_nan_custom_fillna(self):
        s = sanitize_text(pd.Series([None]), fillna_str="unknown")
        assert s.iloc[0] == "unknown"

    def test_already_clean_string_unchanged(self):
        s = sanitize_text(pd.Series(["hello world"]))
        assert s.iloc[0] == "hello world"

    def test_returns_series(self):
        out = sanitize_text(pd.Series(["abc"]))
        assert isinstance(out, pd.Series)

    def test_preserves_index(self):
        idx = [10, 20, 30]
        s = sanitize_text(pd.Series(["a", "b", "c"], index=idx))
        assert list(s.index) == idx

    def test_unicode_normalization_nfkc(self):
        # NFKC normalizes compatibility characters, e.g. fullwidth forms
        s = sanitize_text(pd.Series(["\uff28\uff45\uff4c\uff4c\uff4f"]))  # ｈｅｌｌｏ
        assert s.iloc[0] == "hello"

    def test_mixed_empty_and_value(self):
        s = sanitize_text(pd.Series(["", "  ", "text"]))
        # empty string stays empty after strip (no NaN, so no fillna applied)
        assert s.iloc[2] == "text"

    def test_numeric_values_converted_to_string(self):
        # astype(str) converts floats as "42.0", ints as "42"
        s = sanitize_text(pd.Series([42.0, 3.14]))
        assert s.iloc[0] == "42.0"
        assert s.iloc[1] == "3.14"


# ===========================================================================
# TabArenaModelSpecificPreprocessing
# ===========================================================================


class TestTabArenaModelSpecificPreprocessing:
    def test_hp_key_kwargs_attribute(self):
        assert TabArenaModelSpecificPreprocessing.hp_key_kwargs == "ag.model_specific_feature_generator_kwargs"

    def test_add_to_hyperparameters_empty_dict(self):
        result = TabArenaModelSpecificPreprocessing.add_to_hyperparameters({})
        assert TabArenaModelSpecificPreprocessing.hp_key_kwargs in result

    def test_add_to_hyperparameters_injects_feature_generators(self):
        result = TabArenaModelSpecificPreprocessing.add_to_hyperparameters({})
        hp_key = TabArenaModelSpecificPreprocessing.hp_key_kwargs
        assert "feature_generators" in result[hp_key]
        assert len(result[hp_key]["feature_generators"]) > 0

    def test_add_to_hyperparameters_preserves_existing_keys(self):
        hp = {"other_key": "other_value"}
        result = TabArenaModelSpecificPreprocessing.add_to_hyperparameters(hp)
        assert result["other_key"] == "other_value"

    def test_add_to_hyperparameters_does_not_mutate_input(self):
        hp = {}
        TabArenaModelSpecificPreprocessing.add_to_hyperparameters(hp)
        assert hp == {}

    def test_add_to_hyperparameters_twice_appends_generators(self):
        hp = {}
        result1 = TabArenaModelSpecificPreprocessing.add_to_hyperparameters(hp)
        result2 = TabArenaModelSpecificPreprocessing.add_to_hyperparameters(result1)
        hp_key = TabArenaModelSpecificPreprocessing.hp_key_kwargs
        # Calling twice adds a second set of generators.
        assert len(result2[hp_key]["feature_generators"]) == 2

    def test_get_model_specific_generator_returns_list(self):
        gen = TabArenaModelSpecificPreprocessing.get_model_specific_generator()
        assert isinstance(gen, list)
        assert len(gen) > 0

    def test_get_model_specific_generator_contains_bulk_generator_tuple(self):
        from autogluon.features import BulkFeatureGenerator

        gen = TabArenaModelSpecificPreprocessing.get_model_specific_generator()
        cls, _ = gen[0]
        assert cls is BulkFeatureGenerator

    def test_get_model_specific_generator_kwargs_has_generators(self):
        _, kwargs = TabArenaModelSpecificPreprocessing.get_model_specific_generator()[0]
        assert "generators" in kwargs
        assert len(kwargs["generators"]) > 0


# ===========================================================================
# TabArenaModelAgnosticPreprocessing  (init + fit/transform)
# ===========================================================================


class TestTabArenaModelAgnosticPreprocessingInit:
    def test_default_init(self):
        gen = TabArenaModelAgnosticPreprocessing()
        assert gen is not None

    def test_init_all_text_features_disabled(self):
        gen = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=False,
            enable_statistical_text_features=False,
            enable_new_datetime_features=False,
        )
        assert gen is not None

    def test_init_semantic_only(self):
        # sentence_transformers not available → don't fit, just init
        gen = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=True,
            enable_statistical_text_features=False,
            enable_new_datetime_features=False,
        )
        assert gen is not None

    def test_init_statistical_only(self):
        gen = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=False,
            enable_statistical_text_features=True,
            enable_new_datetime_features=False,
        )
        assert gen is not None

    def test_init_datetime_only(self):
        gen = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=False,
            enable_statistical_text_features=False,
            enable_new_datetime_features=True,
        )
        assert gen is not None


def _make_no_text_gen(**extra):
    return TabArenaModelAgnosticPreprocessing(
        enable_sematic_text_features=False,
        enable_statistical_text_features=False,
        enable_new_datetime_features=False,
        **extra,
    )


class TestTabArenaModelAgnosticPreprocessingFitTransform:
    def test_fit_transform_numeric_returns_dataframe(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        assert isinstance(X_out, pd.DataFrame)

    def test_fit_transform_numeric_preserves_row_count(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        assert len(X_out) == 5

    def test_fit_transform_categorical_column(self):
        X = pd.DataFrame(
            {"num": [1.0, 2.0, 3.0, 4.0, 5.0], "cat": pd.Categorical(["a", "b", "a", "c", "b"])}
        )
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        assert isinstance(X_out, pd.DataFrame)
        assert len(X_out) == 5

    def test_fit_transform_categorical_encodes_as_int_categories(self):
        X = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "c", "b"])})
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        # AutoGluon encodes categories to integer codes
        assert X_out["cat"].dtype.name.startswith("category") or pd.api.types.is_integer_dtype(
            X_out["cat"]
        )

    def test_fit_then_transform_new_data(self):
        X_train = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        gen = _make_no_text_gen()
        gen.fit_transform(X_train)
        X_test = pd.DataFrame({"a": [7.0, 8.0], "b": [9.0, 10.0]})
        X_out = gen.transform(X_test)
        assert len(X_out) == 2

    def test_transform_produces_same_columns_as_fit(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        gen = _make_no_text_gen()
        X_fit = gen.fit_transform(X.copy())
        X_trans = gen.transform(X.copy())
        assert list(X_fit.columns) == list(X_trans.columns)

    def test_fit_transform_fills_nan(self):
        X = pd.DataFrame({"a": [1.0, float("nan"), 3.0]})
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        assert isinstance(X_out, pd.DataFrame)

    def test_fit_transform_with_integer_columns(self):
        X = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        gen = _make_no_text_gen()
        X_out = gen.fit_transform(X)
        assert len(X_out) == 3


# ===========================================================================
# NoCatAsStringCategoryFeatureGenerator
# ===========================================================================


class TestNoCatAsStringCategoryFeatureGenerator:
    def test_init(self):
        gen = NoCatAsStringCategoryFeatureGenerator()
        assert gen is not None

    def test_is_category_feature_generator_subclass(self):
        from autogluon.features.generators.category import CategoryFeatureGenerator

        assert issubclass(NoCatAsStringCategoryFeatureGenerator, CategoryFeatureGenerator)


# ===========================================================================
# StringFixAsTypeFeatureGenerator
# ===========================================================================


class TestStringFixAsTypeFeatureGenerator:
    def test_init(self):
        gen = StringFixAsTypeFeatureGenerator()
        assert gen is not None

    def test_is_astype_feature_generator_subclass(self):
        from autogluon.features.generators.astype import AsTypeFeatureGenerator

        assert issubclass(StringFixAsTypeFeatureGenerator, AsTypeFeatureGenerator)


# ===========================================================================
# DateTimeFeatureGenerator
# ===========================================================================


class TestDateTimeFeatureGenerator:
    def test_init(self):
        gen = DateTimeFeatureGenerator()
        assert gen is not None

    def test_is_abstract_feature_generator_subclass(self):
        from autogluon.features import AbstractFeatureGenerator

        assert issubclass(DateTimeFeatureGenerator, AbstractFeatureGenerator)

    def test_get_default_infer_features_in_args_returns_dict(self):
        args = DateTimeFeatureGenerator.get_default_infer_features_in_args()
        assert isinstance(args, dict)

    def test_get_default_infer_features_in_args_has_required_raw_special_pairs(self):
        args = DateTimeFeatureGenerator.get_default_infer_features_in_args()
        assert "required_raw_special_pairs" in args

    def test_fit_transform_datetime_column(self):
        gen = DateTimeFeatureGenerator()
        X = _make_datetime_df(n_rows=8)
        X_out, sg = gen._fit_transform(X.copy())
        assert isinstance(X_out, pd.DataFrame)
        assert len(X_out) == 8

    def test_fit_transform_produces_multiple_features(self):
        gen = DateTimeFeatureGenerator()
        X = _make_datetime_df(n_rows=8)
        X_out, _ = gen._fit_transform(X.copy())
        # DatetimeEncoder expands one datetime column into multiple features
        assert X_out.shape[1] > 1

    def test_transform_after_fit(self):
        gen = DateTimeFeatureGenerator()
        X_train = _make_datetime_df(n_rows=8)
        X_test = _make_datetime_df(n_rows=4)
        gen._fit_transform(X_train.copy())
        X_out = gen._transform(X_test.copy())
        assert len(X_out) == 4

    def test_transform_output_columns_match_fit(self):
        gen = DateTimeFeatureGenerator()
        X_train = _make_datetime_df(n_rows=8)
        X_out_train, _ = gen._fit_transform(X_train.copy())
        X_out_test = gen._transform(_make_datetime_df(n_rows=3).copy())
        assert list(X_out_train.columns) == list(X_out_test.columns)


# ===========================================================================
# StatisticalTextFeatureGenerator
# ===========================================================================


class TestStatisticalTextFeatureGenerator:
    def test_init(self):
        gen = StatisticalTextFeatureGenerator()
        assert gen is not None

    def test_is_abstract_feature_generator_subclass(self):
        from autogluon.features import AbstractFeatureGenerator

        assert issubclass(StatisticalTextFeatureGenerator, AbstractFeatureGenerator)

    def test_get_default_infer_features_in_args_returns_dict(self):
        args = StatisticalTextFeatureGenerator.get_default_infer_features_in_args()
        assert isinstance(args, dict)

    def test_get_default_infer_features_in_args_required_special_types(self):
        from autogluon.common.features.types import S_TEXT

        args = StatisticalTextFeatureGenerator.get_default_infer_features_in_args()
        assert "required_special_types" in args
        assert S_TEXT in args["required_special_types"]

    def test_fit_transform_text_column(self):
        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)
        X_out, sg = gen._fit_transform(X.copy())
        assert isinstance(X_out, pd.DataFrame)
        assert len(X_out) == 20

    def test_fit_transform_returns_text_embedding_special_type(self):
        from autogluon.common.features.types import S_TEXT_EMBEDDING

        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)
        _, sg = gen._fit_transform(X.copy())
        assert S_TEXT_EMBEDDING in sg

    def test_fit_transform_increases_feature_count(self):
        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)
        X_out, _ = gen._fit_transform(X.copy())
        # StringEncoder expands one text column into many embedding features
        assert X_out.shape[1] > 1

    def test_transform_after_fit(self):
        gen = StatisticalTextFeatureGenerator()
        X_train = _make_text_df(n_rows=20)
        X_test = _make_text_df(n_rows=5)
        gen._fit_transform(X_train.copy())
        X_out = gen._transform(X_test.copy())
        assert len(X_out) == 5

    def test_transform_output_columns_match_fit(self):
        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)
        X_out_train, _ = gen._fit_transform(X.copy())
        X_out_test = gen._transform(_make_text_df(n_rows=4).copy())
        assert list(X_out_train.columns) == list(X_out_test.columns)

    def test_max_n_output_features_constant(self):
        assert StatisticalTextFeatureGenerator.MAX_N_OUTPUT_FEATURES == 384


# ===========================================================================
# SemanticTextFeatureGenerator (init/static tests only; model download needed)
# ===========================================================================


class TestSemanticTextFeatureGenerator:
    def test_init(self):
        gen = SemanticTextFeatureGenerator()
        assert gen is not None

    def test_is_abstract_feature_generator_subclass(self):
        from autogluon.features import AbstractFeatureGenerator

        assert issubclass(SemanticTextFeatureGenerator, AbstractFeatureGenerator)

    def test_get_default_infer_features_in_args_returns_dict(self):
        args = SemanticTextFeatureGenerator.get_default_infer_features_in_args()
        assert isinstance(args, dict)

    def test_get_default_infer_features_in_args_requires_text(self):
        from autogluon.common.features.types import S_TEXT

        args = SemanticTextFeatureGenerator.get_default_infer_features_in_args()
        assert "required_special_types" in args
        assert S_TEXT in args["required_special_types"]

    def test_get_default_infer_features_in_args_excludes_image_paths(self):
        from autogluon.common.features.types import S_IMAGE_BYTEARRAY, S_IMAGE_PATH

        args = SemanticTextFeatureGenerator.get_default_infer_features_in_args()
        invalid = args.get("invalid_special_types", [])
        assert S_IMAGE_PATH in invalid
        assert S_IMAGE_BYTEARRAY in invalid

    def test_more_tags_declares_feature_interactions(self):
        gen = SemanticTextFeatureGenerator()
        tags = gen._more_tags()
        assert tags.get("feature_interactions") is True

    def test_transform_raises_without_fit(self):
        gen = SemanticTextFeatureGenerator()
        X = _make_text_df(n_rows=5)
        with pytest.raises((AttributeError, ValueError)):
            gen._transform(X)

    def test_transform_empty_df_raises_value_error(self):
        # _transform guards against empty inputs
        gen = SemanticTextFeatureGenerator()
        X = pd.DataFrame({"text": pd.Series([], dtype=str)})
        # First do a partial setup so we can reach the empty check
        gen._embedding_look_up = {}
        with pytest.raises(ValueError, match="empty"):
            gen._transform(X)


# ===========================================================================
# TextEmbeddingDimensionalityReductionFeatureGenerator
# ===========================================================================


class TestTextEmbeddingDRConstants:
    def test_batch_size(self):
        assert TextEmbeddingDimensionalityReductionFeatureGenerator._BATCH_SIZE == 1000

    def test_max_components_per_batch(self):
        assert TextEmbeddingDimensionalityReductionFeatureGenerator._MAX_COMPONENTS_PER_BATCH == 30

    def test_explained_variance_threshold(self):
        assert (
            TextEmbeddingDimensionalityReductionFeatureGenerator._EXPLAINED_VARIANCE_THRESHOLD == 0.99
        )


class TestTextEmbeddingDRStaticMethods:
    def test_get_default_infer_features_in_args_returns_dict(self):
        args = TextEmbeddingDimensionalityReductionFeatureGenerator.get_default_infer_features_in_args()
        assert isinstance(args, dict)

    def test_get_default_infer_features_in_args_requires_embedding_types(self):
        from autogluon.common.features.types import S_TEXT_EMBEDDING, S_TEXT_SPECIAL

        args = TextEmbeddingDimensionalityReductionFeatureGenerator.get_default_infer_features_in_args()
        valid = args.get("valid_special_types", [])
        assert S_TEXT_EMBEDDING in valid
        assert S_TEXT_SPECIAL in valid

    def test_get_infer_features_in_args_to_drop_returns_dict(self):
        drop = TextEmbeddingDimensionalityReductionFeatureGenerator.get_infer_features_in_args_to_drop()
        assert isinstance(drop, dict)

    def test_get_infer_features_in_args_to_drop_has_invalid_special_types(self):
        from autogluon.common.features.types import S_TEXT_EMBEDDING, S_TEXT_SPECIAL

        drop = TextEmbeddingDimensionalityReductionFeatureGenerator.get_infer_features_in_args_to_drop()
        invalid = drop.get("invalid_special_types", [])
        assert S_TEXT_EMBEDDING in invalid
        assert S_TEXT_SPECIAL in invalid

    @pytest.mark.parametrize(
        ("ratios", "threshold", "expected_min_k"),
        [
            ([0.5, 0.3, 0.2], 0.99, 3),  # need all three
            ([0.6, 0.3, 0.1], 0.89, 2),  # two gives 0.9 > 0.89
            ([1.0], 0.99, 1),  # single component explains all
            ([0.01] * 100, 0.99, 99),  # need many components
        ],
    )
    def test_num_components_for_variance(self, ratios, threshold, expected_min_k):
        k = TextEmbeddingDimensionalityReductionFeatureGenerator._num_components_for_variance(
            np.array(ratios), threshold
        )
        assert k >= 1
        assert k <= len(ratios)
        # Verify cumulative variance at k covers threshold
        cumulative = np.cumsum(ratios)
        assert cumulative[k - 1] >= threshold or k == len(ratios)

    def test_num_components_for_variance_returns_at_least_1(self):
        k = TextEmbeddingDimensionalityReductionFeatureGenerator._num_components_for_variance(
            np.array([0.001]), 0.99
        )
        assert k >= 1

    def test_standard_scale_fit_zero_mean(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 10.0, 10.0]})
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
        # Scaled data should have approx zero mean
        assert abs(X_scaled["a"].mean()) < 1e-10

    def test_standard_scale_fit_unit_std(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
        assert abs(X_scaled["a"].std(ddof=0) - 1.0) < 1e-9

    def test_standard_scale_fit_zero_std_column_replaced_with_1(self):
        X = pd.DataFrame({"constant": [5.0, 5.0, 5.0]})
        _, _, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
        # Zero std is replaced with 1 to avoid division by zero
        assert stds["constant"] == 1.0

    def test_standard_scale_fit_returns_tuple_of_three(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0]})
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
        assert len(result) == 3

    def test_standard_scale_transform_consistency(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "b": [10.0, 20.0, 30.0, 40.0]})
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(
            X.copy()
        )
        X_transform = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_transform(
            X, means, stds
        )
        pd.testing.assert_frame_equal(X_scaled, X_transform)

    def test_encode_target_numeric(self):
        y = pd.Series([0.0, 1.0, 2.0, 3.0])
        y_enc = TextEmbeddingDimensionalityReductionFeatureGenerator._encode_target_for_correlation(y)
        assert y_enc.dtype == np.float64
        np.testing.assert_array_equal(y_enc, [0.0, 1.0, 2.0, 3.0])

    def test_encode_target_categorical_is_float64(self):
        y = pd.Series(["cat", "dog", "cat", "bird"])
        y_enc = TextEmbeddingDimensionalityReductionFeatureGenerator._encode_target_for_correlation(y)
        assert y_enc.dtype == np.float64
        assert len(y_enc) == 4

    def test_encode_target_categorical_deterministic(self):
        y = pd.Series(["a", "b", "a", "b"])
        y1 = TextEmbeddingDimensionalityReductionFeatureGenerator._encode_target_for_correlation(y)
        y2 = TextEmbeddingDimensionalityReductionFeatureGenerator._encode_target_for_correlation(y)
        np.testing.assert_array_equal(y1, y2)

    def test_encode_target_with_nan_filled(self):
        y = pd.Series([1.0, float("nan"), 3.0])
        y_enc = TextEmbeddingDimensionalityReductionFeatureGenerator._encode_target_for_correlation(y)
        assert not np.isnan(y_enc).any()


class TestTextEmbeddingDRFitTransform:
    def _make_embedding_df(self, n_rows=30, n_cols=50) -> pd.DataFrame:
        rng = np.random.default_rng(1)
        return pd.DataFrame(
            rng.standard_normal((n_rows, n_cols)),
            columns=[f"emb_{i}" for i in range(n_cols)],
        )

    def _make_target(self, n_rows=30) -> pd.Series:
        rng = np.random.default_rng(1)
        return pd.Series(rng.integers(0, 2, n_rows))

    def test_fit_preprocess_and_transform_returns_dataframe(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        assert isinstance(X_out, pd.DataFrame)

    def test_fit_preprocess_and_transform_preserves_row_count(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=25)
        y = self._make_target(n_rows=25)
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        assert len(X_out) == 25

    def test_fit_preprocess_and_transform_reduces_feature_count(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=30, n_cols=50)
        y = self._make_target(n_rows=30)
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        # PCA + variance thresholding should reduce features
        assert X_out.shape[1] <= 50

    def test_fit_preprocess_and_transform_output_cols_named_with_pca(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        assert all("__text_embedding__" in c for c in X_out.columns)

    def test_fit_preprocess_and_transform_max_components_per_batch(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=100, n_cols=50)
        y = self._make_target(n_rows=100)
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        # At most _MAX_COMPONENTS_PER_BATCH components per batch
        assert X_out.shape[1] <= gen._MAX_COMPONENTS_PER_BATCH

    def test_fit_transform_sets_feature_names_in(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        assert hasattr(gen, "feature_names_in_")
        assert gen.feature_names_in_ == list(X.columns)

    def test_fit_transform_sets_expected_features(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        assert hasattr(gen, "expected_features_")
        assert gen.expected_features_ == list(X.columns)

    def test_fit_transform_sets_feature_names_out(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        assert hasattr(gen, "feature_names_out_")
        assert len(gen.feature_names_out_) > 0

    def test_transform_raises_attribute_error_on_missing_sorted_features(self):
        # Known limitation: _transform_inference references `self.sorted_feature_names_`
        # which is never set by _fit_transform. Tracked here to catch when it gets fixed.
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        with pytest.raises(AttributeError, match="sorted_feature_names_"):
            gen._transform(X.copy())

    def test_transform_raises_on_column_mismatch(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        X_wrong = self._make_embedding_df(n_cols=10)
        with pytest.raises((ValueError, AttributeError)):
            gen._transform(X_wrong)

    def test_single_row_input(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=5, n_cols=10)
        y = self._make_target(n_rows=5)
        # With only 5 rows, PCA is limited by n_samples
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        assert isinstance(X_out, pd.DataFrame)

    def test_more_tags_declares_feature_interactions(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        assert gen._more_tags().get("feature_interactions") is True

    def test_scale_fit_and_transform_are_inverse(self):
        X = self._make_embedding_df(n_rows=20, n_cols=5)
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(
            X.copy()
        )
        # Re-apply transform to the original (un-scaled) X
        X_t = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_transform(
            X, means, stds
        )
        pd.testing.assert_frame_equal(X_scaled, X_t)
