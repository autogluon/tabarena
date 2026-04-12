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
    StringFixAsTypeFeatureGenerator,
)
from tabarena.benchmark.preprocessing.model_specific_default_preprocessing import (
    NoCatAsStringCategoryFeatureGenerator,
)
from tabarena.benchmark.preprocessing.text_feature_generators import (
    SemanticTextFeatureGenerator,
    StatisticalTextFeatureGenerator,
    TabArenaDefaultTextEncoder,
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
    phrases = [
        "hello world",
        "foo bar baz",
        "quick brown fox",
        "the lazy dog",
        "test text",
    ]
    return pd.DataFrame({"text": [phrases[i % len(phrases)] for i in range(n_rows)]})


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

    def test_control_chars_removed(self):
        # \x01 (SOH) is a non-printable control char and should be removed.
        s = sanitize_text(pd.Series(["hello\x01world", "foo\x07bar"]))
        assert s.iloc[0] == "helloworld"
        assert s.iloc[1] == "foobar"

    def test_whitespace_control_chars_preserved(self):
        # Tab \x09 and newline \x0a are whitespace and should NOT be removed
        # (they get collapsed to a single space by the \s+ replace).
        s = sanitize_text(pd.Series(["foo\x09bar"]))
        assert s.iloc[0] == "foo bar"

    def test_del_char_removed(self):
        # \x7f (DEL) is a control char and should be stripped.
        s = sanitize_text(pd.Series(["abc\x7fdef"]))
        assert s.iloc[0] == "abcdef"


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
            {
                "num": [1.0, 2.0, 3.0, 4.0, 5.0],
                "cat": pd.Categorical(["a", "b", "a", "c", "b"]),
            }
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
        assert X_out["cat"].dtype.name.startswith("category") or pd.api.types.is_integer_dtype(X_out["cat"])

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

    def test_fit_transform_renames_dot_columns(self):
        gen = _make_no_text_gen()
        X = pd.DataFrame({"a.b": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]})
        X_out = gen.fit_transform(X.copy())
        assert "a_b" in X_out.columns
        assert "a.b" not in X_out.columns

    def test_fit_transform_leaves_clean_columns_unchanged(self):
        gen = _make_no_text_gen()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b_c": [4.0, 5.0, 6.0]})
        X_out = gen.fit_transform(X.copy())
        assert "a" in X_out.columns
        assert "b_c" in X_out.columns

    def test_fit_transform_multiple_dots_all_replaced(self):
        gen = _make_no_text_gen()
        X = pd.DataFrame({"a.b.c": [1.0, 2.0, 3.0]})
        X_out = gen.fit_transform(X.copy())
        assert "a_b_c" in X_out.columns

    def test_transform_also_renames_dot_columns(self):
        gen = _make_no_text_gen()
        X_train = pd.DataFrame({"a.b": [1.0, 2.0, 3.0], "c": [4.0, 5.0, 6.0]})
        gen.fit_transform(X_train.copy())
        X_test = pd.DataFrame({"a.b": [7.0, 8.0], "c": [9.0, 10.0]})
        X_out = gen.transform(X_test)
        assert "a_b" in X_out.columns
        assert "a.b" not in X_out.columns

    def test_no_dot_columns_produces_empty_rename_map(self):
        gen = _make_no_text_gen()
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [4.0, 5.0, 6.0]})
        gen.fit_transform(X.copy())
        assert gen._dot_rename_map_ == {}


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
# TabArenaNoCatAsStringCategoryFeatureGenerator
# ===========================================================================


class TestNoCatAsStringCategoryFeatureGeneratorUnseenHandling:
    """Tests for the unseen-category preservation behaviour."""

    def _fit_gen(self, X_train: pd.DataFrame) -> NoCatAsStringCategoryFeatureGenerator:
        gen = NoCatAsStringCategoryFeatureGenerator()
        gen.fit_transform(X_train.copy())
        return gen

    # ------------------------------------------------------------------
    # Inheritance / init
    # ------------------------------------------------------------------

    def test_init(self):
        gen = NoCatAsStringCategoryFeatureGenerator()
        assert gen is not None

    # ------------------------------------------------------------------
    # Known categories still work correctly
    # ------------------------------------------------------------------

    def test_known_categories_no_nan(self):
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0

    def test_known_categories_values_preserved(self):
        """Known category values must be unchanged after transform."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "b"])})
        X_out = gen.transform(X_test.copy())
        vals = X_out["cat"].astype(object).tolist()
        assert vals == ["a", "b"]

    # ------------------------------------------------------------------
    # Unseen category → original value kept (not NaN)
    # ------------------------------------------------------------------

    def test_unseen_category_not_nan(self):
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "d"])})  # 'd' is new
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0, "Unseen category 'd' was converted to NaN"

    def test_unseen_category_value_preserved(self):
        """The original unseen value must appear verbatim in the output."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "NOVEL"])})
        X_out = gen.transform(X_test.copy())
        vals = X_out["cat"].astype(object).tolist()
        assert vals[1] == "NOVEL", f"Unseen value should be 'NOVEL', got {vals[1]}"
        assert vals[0] == "a"

    def test_multiple_distinct_unseen_values_all_preserved(self):
        """Each distinct unseen value must survive unchanged."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["p", "q", "p", "q"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["UNSEEN1", "UNSEEN2", "UNSEEN3"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0
        vals = X_out["cat"].astype(object).tolist()
        assert vals == ["UNSEEN1", "UNSEEN2", "UNSEEN3"]

    def test_nan_stays_nan_unseen_value_preserved(self):
        """Original NaN must remain NaN; unseen non-NaN must keep its value."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Series([None, "a", "NEW"], dtype=object)})
        X_out = gen.transform(X_test.copy())
        assert pd.isna(X_out["cat"].iloc[0]), "Original NaN must stay NaN"
        assert not pd.isna(X_out["cat"].iloc[2]), "Unseen 'NEW' must not be NaN"
        assert X_out["cat"].astype(object).iloc[2] == "NEW"

    def test_mixed_known_unseen_and_nan(self):
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Series(["a", None, "b", "UNSEEN"], dtype=object)})
        X_out = gen.transform(X_test.copy())
        vals = X_out["cat"].astype(object).tolist()
        assert vals[0] == "a"
        assert pd.isna(vals[1])
        assert vals[2] == "b"
        assert vals[3] == "UNSEEN"

    # ------------------------------------------------------------------
    # Multiple columns
    # ------------------------------------------------------------------

    def test_multiple_columns_unseen_in_one(self):
        X_train = pd.DataFrame(
            {
                "c1": pd.Categorical(["a", "b", "a", "b"]),
                "c2": pd.Categorical(["x", "y", "x", "y"]),
            }
        )
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame(
            {
                "c1": pd.Categorical(["a", "NEW_C1"]),
                "c2": pd.Categorical(["x", "y"]),
            }
        )
        X_out = gen.transform(X_test.copy())
        assert X_out["c1"].isna().sum() == 0
        assert X_out["c2"].isna().sum() == 0
        assert X_out["c1"].astype(object).iloc[1] == "NEW_C1"


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
# StringFixAsTypeFeatureGenerator – categorical dtype special cases
# ===========================================================================


class TestStringFixAsTypeFeatureGeneratorCategoricals:
    """Tests for the fix that prevents unknown categories from becoming NaN at test time."""

    def _fit_gen(self, X_train: pd.DataFrame) -> StringFixAsTypeFeatureGenerator:
        gen = StringFixAsTypeFeatureGenerator()
        gen.fit_transform(X_train.copy())
        return gen

    # ------------------------------------------------------------------
    # Core regression: unknown categories must NOT become NaN
    # ------------------------------------------------------------------

    def test_unknown_category_at_test_time_not_nan(self):
        """A category value unseen during training must not be silently mapped to NaN."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "a", "c", "b"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "d", "b"])})  # 'd' is new
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0, "Unknown category 'd' was converted to NaN"

    def test_unknown_category_value_present_in_output(self):
        """The actual unknown value must appear in the transformed output.

        Requires 3+ training categories so the column is stored as CategoricalDtype
        (not bool/int8) in _type_map_real_opt, which is the path our fix protects.
        """
        X_train = pd.DataFrame({"cat": pd.Categorical(["x", "y", "w"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["x", "z"])})  # 'z' is new
        X_out = gen.transform(X_test.copy())
        values = set(X_out["cat"].astype(object).tolist())
        assert "z" in values, "Unknown category 'z' missing from output"

    def test_multiple_unknown_categories_all_preserved(self):
        """All novel category values must survive the transform.

        Requires 3+ training categories so the column is stored as CategoricalDtype
        (not bool/int8) in _type_map_real_opt, which is the path our fix protects.
        """
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "base"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["c", "d", "e"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0
        values = set(X_out["cat"].astype(object).tolist())
        assert values == {"c", "d", "e"}

    def test_mix_of_known_and_unknown_categories_no_nan(self):
        """Mix of seen and unseen category values should both survive."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "b", "NEW"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0
        values = set(X_out["cat"].astype(object).tolist())
        assert "NEW" in values

    # ------------------------------------------------------------------
    # Known categories still work correctly (regression guard)
    # ------------------------------------------------------------------

    def test_known_categories_at_test_time_no_nan(self):
        """Values seen during training must not become NaN."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0

    def test_known_categories_at_test_time_values_preserved(self):
        X_train = pd.DataFrame({"cat": pd.Categorical(["p", "q", "r"])})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"cat": pd.Categorical(["p", "r"])})
        X_out = gen.transform(X_test.copy())
        values = set(X_out["cat"].astype(object).tolist())
        assert values == {"p", "r"}

    # ------------------------------------------------------------------
    # Ordered categoricals
    # ------------------------------------------------------------------

    def test_ordered_categorical_unknown_value_not_nan(self):
        """Unknown values in an ordered categorical must not become NaN.

        The test data is passed as object dtype to avoid pandas itself NaN-ifying
        'extreme' when constructing a Categorical restricted to training categories.
        """
        dtype = pd.CategoricalDtype(categories=["low", "med", "high"], ordered=True)
        X_train = pd.DataFrame({"level": pd.Categorical(["low", "med", "high"], dtype=dtype)})
        gen = self._fit_gen(X_train)
        # Pass as object so pandas does not NaN 'extreme' at DataFrame construction time.
        X_test = pd.DataFrame({"level": pd.Series(["low", "extreme"], dtype=object)})
        X_out = gen.transform(X_test.copy())
        assert X_out["level"].isna().sum() == 0

    # ------------------------------------------------------------------
    # Object-typed column at test time (not yet categorical)
    # ------------------------------------------------------------------

    def test_object_column_with_unknown_value_not_nan(self):
        """If test data arrives as object dtype (not categorical), unknown values must survive."""
        X_train = pd.DataFrame({"cat": pd.Categorical(["a", "b", "c"])})
        gen = self._fit_gen(X_train)
        # Deliver as plain object, not Categorical
        X_test = pd.DataFrame({"cat": pd.Series(["a", "b", "NEW"], dtype=object)})
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0

    # ------------------------------------------------------------------
    # Multiple categorical columns
    # ------------------------------------------------------------------

    def test_multiple_categorical_columns_unknown_values_preserved(self):
        X_train = pd.DataFrame(
            {
                "c1": pd.Categorical(["a", "b"]),
                "c2": pd.Categorical(["x", "y"]),
            }
        )
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame(
            {
                "c1": pd.Categorical(["a", "NOVEL"]),
                "c2": pd.Categorical(["x", "ALSO_NEW"]),
            }
        )
        X_out = gen.transform(X_test.copy())
        assert X_out["c1"].isna().sum() == 0
        assert X_out["c2"].isna().sum() == 0

    # ------------------------------------------------------------------
    # Non-categorical columns are unaffected
    # ------------------------------------------------------------------

    def test_float_column_unchanged_by_categorical_fix(self):
        """Float columns must still pass through correctly alongside categoricals."""
        X_train = pd.DataFrame(
            {
                "num": [1.0, 2.0, 3.0],
                "cat": pd.Categorical(["a", "b", "c"]),
            }
        )
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame(
            {
                "num": [4.0, 5.0],
                "cat": pd.Categorical(["a", "NEW"]),
            }
        )
        X_out = gen.transform(X_test.copy())
        assert X_out["cat"].isna().sum() == 0
        assert X_out["num"].isna().sum() == 0

    # ------------------------------------------------------------------
    # Int columns: NaN at test time but not at train time → imputed to 0
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Bool columns with unseen values at test time
    #
    # Bool encoding always applies: true_val → 1, everything else → 0.
    # Unseen values are mapped to 0 (False) and a warning is logged.
    # The column stays in _bool_features and keeps its int8 dtype.
    # ------------------------------------------------------------------

    def test_bool_col_unseen_value_mapped_to_false(self):
        """An unseen value in a bool column must be mapped to 0 (False)."""
        X_train = pd.DataFrame({"col": pd.Categorical(["yes", "no", "yes", "no"])})
        gen = StringFixAsTypeFeatureGenerator()
        gen.fit_transform(X_train.copy())
        assert "col" in gen._bool_features

        X_test = pd.DataFrame({"col": pd.Categorical(["yes", "no", "maybe"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["col"].dtype == np.int8
        assert X_out["col"].iloc[0] == 1  # 'yes' (true_val) → 1
        assert X_out["col"].iloc[1] == 0  # 'no' (false_val) → 0
        assert X_out["col"].iloc[2] == 0  # 'maybe' (unseen) → 0

    def test_bool_int_col_unseen_value_mapped_to_false(self):
        """An unseen integer in a bool int column (0/1) must be mapped to 0."""
        X_train = pd.DataFrame({"b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)
        X_out = gen.transform(pd.DataFrame({"b": [0, 1, 2]}))
        assert X_out["b"].dtype == np.int8
        assert list(X_out["b"]) == [0, 1, 0]  # 2 is unseen → 0

    def test_bool_int_col_multiple_unseen_values_all_map_to_false(self):
        """All unseen integer values must be mapped to 0."""
        X_train = pd.DataFrame({"b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)
        X_out = gen.transform(pd.DataFrame({"b": [0, 1, 5, 7, 9]}))
        assert list(X_out["b"]) == [0, 1, 0, 0, 0]

    def test_bool_string_col_unseen_value_mapped_to_false(self):
        """An unseen string in a bool string column must be mapped to 0."""
        X_train = pd.DataFrame({"b": pd.Categorical(["yes", "no", "yes", "no"])})
        gen = self._fit_gen(X_train)
        X_out = gen.transform(pd.DataFrame({"b": pd.Categorical(["yes", "no", "maybe"])}))
        assert X_out["b"].dtype == np.int8
        assert X_out["b"].iloc[0] == 1  # 'yes' → 1
        assert X_out["b"].iloc[1] == 0  # 'no' → 0
        assert X_out["b"].iloc[2] == 0  # 'maybe' → 0

    def test_bool_col_stays_in_bool_features_after_unseen(self):
        """A bool column with unseen values must remain in _bool_features."""
        X_train = pd.DataFrame({"b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)
        assert "b" in gen._bool_features

        gen.transform(pd.DataFrame({"b": [0, 1, 2]}))
        assert "b" in gen._bool_features, "Column must remain a bool feature"

    def test_bool_col_second_transform_still_bool_encodes(self):
        """A second transform call must still apply bool encoding."""
        X_train = pd.DataFrame({"b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)

        gen.transform(pd.DataFrame({"b": [0, 1, 2]}))

        X_out = gen.transform(pd.DataFrame({"b": [0, 1, 3]}))
        assert X_out["b"].dtype == np.int8
        assert list(X_out["b"]) == [0, 1, 0]  # 3 is unseen → 0

    def test_bool_col_unseen_other_bool_col_unaffected(self):
        """A sibling bool column without unseen values must still be bool-encoded normally."""
        X_train = pd.DataFrame({"a": [0, 1, 0, 1], "b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)

        X_test = pd.DataFrame({"a": [0, 1, 2], "b": [0, 1, 0]})
        X_out = gen.transform(X_test.copy())
        # 'a' gained unseen → still int8, unseen mapped to 0
        assert X_out["a"].dtype == np.int8
        assert list(X_out["a"]) == [0, 1, 0]
        # 'b' no unseen → normal bool-encoded 0/1
        assert X_out["b"].dtype == np.int8
        assert set(X_out["b"].tolist()).issubset({0, 1})

    def test_bool_col_without_unseen_values_still_bool_encoded(self):
        """When no new categories appear the bool-encoding path must still run normally."""
        X_train = pd.DataFrame({"col": pd.Categorical(["yes", "no", "yes", "no"])})
        gen = StringFixAsTypeFeatureGenerator()
        gen.fit_transform(X_train.copy())
        X_test = pd.DataFrame({"col": pd.Categorical(["yes", "no", "yes"])})
        X_out = gen.transform(X_test.copy())
        assert X_out["col"].isna().sum() == 0
        assert set(X_out["col"].tolist()).issubset({0, 1})

    def test_bool_col_with_nan_at_test_time(self):
        """NaN in a bool column with unseen values is imputed to 0, consistent
        with how all int columns handle test-time NaN.
        """
        X_train = pd.DataFrame({"b": [0, 1, 0, 1]})
        gen = self._fit_gen(X_train)
        X_out = gen.transform(pd.DataFrame({"b": [0, 1, 2, None]}))
        assert X_out["b"].iloc[3] == 0
        assert X_out["b"].dtype == np.int8

    def test_int_column_with_nan_at_test_time_imputed_to_zero(self):
        """Int features that were never NaN at train time must be imputed to 0 at test time."""
        X_train = pd.DataFrame({"val": [1, 2, 3, 4, 5]})
        gen = self._fit_gen(X_train)
        X_test = pd.DataFrame({"val": [1, None, 3]})
        X_out = gen.transform(X_test.copy())
        assert X_out["val"].isna().sum() == 0
        assert X_out["val"].iloc[1] == 0


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
        X_out, _sg = gen._fit_transform(X.copy())
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
        X_out, _sg = gen._fit_transform(X.copy())
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
        assert StatisticalTextFeatureGenerator.MAX_N_OUTPUT_FEATURES == 32

    def test_output_columns_prefixed_with_source_column(self):
        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)  # column named "text"
        X_out, _ = gen._fit_transform(X.copy())
        # TableVectorizer produces "text_0", "text_1", ... which get renamed to "text.0", "text.1", ...
        assert all(c.startswith("text.") for c in X_out.columns)

    def test_output_column_format_uses_dot_separator(self):
        gen = StatisticalTextFeatureGenerator()
        X = _make_text_df(n_rows=20)
        X_out, _ = gen._fit_transform(X.copy())
        # All columns should match "{col}.{integer}" format
        import re

        assert all(re.match(r"^text\.\d+$", c) for c in X_out.columns)

    def test_two_columns_each_prefixed_correctly(self):
        gen = StatisticalTextFeatureGenerator()
        phrases = [
            "hello world",
            "foo bar baz",
            "quick brown fox",
            "test text",
            "sample",
        ]
        n = 20
        X = pd.DataFrame(
            {
                "col_a": [phrases[i % len(phrases)] for i in range(n)],
                "col_b": [phrases[(i + 1) % len(phrases)] for i in range(n)],
            }
        )
        X_out, _ = gen._fit_transform(X.copy())
        a_cols = [c for c in X_out.columns if c.startswith("col_a.")]
        b_cols = [c for c in X_out.columns if c.startswith("col_b.")]
        assert len(a_cols) > 0
        assert len(b_cols) > 0
        assert len(a_cols) + len(b_cols) == X_out.shape[1]


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


class TestSemanticTextFeatureGeneratorCacheRoundTrip:
    """End-to-end: fit_transform → save cache to parquet → clear → load cache → transform → compare."""

    EMB_DIM = 32

    @pytest.fixture(autouse=True)
    def _clean_embedding_cache(self):
        """Isolate the class-level cache for each test."""
        saved = dict(SemanticTextFeatureGenerator._embedding_look_up)
        SemanticTextFeatureGenerator._embedding_look_up.clear()
        yield
        SemanticTextFeatureGenerator._embedding_look_up.clear()
        SemanticTextFeatureGenerator._embedding_look_up.update(saved)

    @staticmethod
    def _deterministic_embeddings(texts: list[str]) -> np.ndarray:
        """Hash-based deterministic 32-dim embeddings."""
        import hashlib

        embs = []
        for t in texts:
            seed = int(hashlib.md5(t.encode()).hexdigest(), 16) % (2**31)
            rng = np.random.RandomState(seed)
            emb = rng.randn(32).astype(np.float32)
            emb /= np.linalg.norm(emb)
            embs.append(emb)
        return np.vstack(embs)

    def test_save_load_roundtrip_produces_identical_output(self, tmp_path, monkeypatch):
        """Full pipeline: fit_transform, save cache, clear, load cache, transform, compare."""
        monkeypatch.setattr(TabArenaDefaultTextEncoder, "get_default_encoder", lambda: None)
        monkeypatch.setattr(
            TabArenaDefaultTextEncoder,
            "encode_texts",
            lambda *, texts, encoder_model: self._deterministic_embeddings(texts),
        )

        X = _make_text_df(n_rows=20)
        gen = SemanticTextFeatureGenerator()

        # 1. fit_transform populates _embedding_look_up and produces output
        X_out_fit, _type_map = gen._fit_transform(X)
        assert not X_out_fit.empty
        cache = dict(SemanticTextFeatureGenerator._embedding_look_up)
        assert len(cache) > 0

        # 2. Save cache to parquet
        cache_path = tmp_path / "text_cache.parquet"
        SemanticTextFeatureGenerator.save_embedding_cache(cache=cache, path=cache_path)
        assert cache_path.exists()

        # 3. Clear class-level cache
        SemanticTextFeatureGenerator._embedding_look_up.clear()
        assert len(SemanticTextFeatureGenerator._embedding_look_up) == 0

        # 4. Load cache from disk
        loaded = SemanticTextFeatureGenerator.load_embedding_cache(cache_path)
        SemanticTextFeatureGenerator._embedding_look_up.update(loaded)
        assert set(loaded.keys()) == set(cache.keys())

        # 5. Transform with loaded cache (no encoding happens)
        X_out_cached = gen._transform(X)

        # 6. Output must be identical
        pd.testing.assert_frame_equal(X_out_fit, X_out_cached)

    def test_loaded_embeddings_match_original_values(self, tmp_path, monkeypatch):
        """Verify that individual embedding vectors survive the parquet round-trip."""
        monkeypatch.setattr(TabArenaDefaultTextEncoder, "get_default_encoder", lambda: None)
        monkeypatch.setattr(
            TabArenaDefaultTextEncoder,
            "encode_texts",
            lambda *, texts, encoder_model: self._deterministic_embeddings(texts),
        )

        X = _make_text_df(n_rows=20)
        gen = SemanticTextFeatureGenerator()
        gen._fit_transform(X)

        original_cache = {k: v.copy() for k, v in SemanticTextFeatureGenerator._embedding_look_up.items()}

        cache_path = tmp_path / "emb_cache.parquet"
        SemanticTextFeatureGenerator.save_embedding_cache(cache=original_cache, path=cache_path)
        loaded_cache = SemanticTextFeatureGenerator.load_embedding_cache(cache_path)

        for key in original_cache:
            np.testing.assert_array_almost_equal(loaded_cache[key], original_cache[key], decimal=5)

    def test_cache_roundtrip_with_unseen_data_at_transform(self, tmp_path, monkeypatch):
        """Load cache from one fit, then transform data that includes new unseen text values."""
        monkeypatch.setattr(TabArenaDefaultTextEncoder, "get_default_encoder", lambda: None)
        monkeypatch.setattr(
            TabArenaDefaultTextEncoder,
            "encode_texts",
            lambda *, texts, encoder_model: self._deterministic_embeddings(texts),
        )

        X_train = _make_text_df(n_rows=20)
        gen = SemanticTextFeatureGenerator()
        gen._fit_transform(X_train)

        cache_path = tmp_path / "partial_cache.parquet"
        SemanticTextFeatureGenerator.save_embedding_cache(
            cache=SemanticTextFeatureGenerator._embedding_look_up, path=cache_path
        )

        # Clear and reload
        SemanticTextFeatureGenerator._embedding_look_up.clear()
        loaded = SemanticTextFeatureGenerator.load_embedding_cache(cache_path)
        SemanticTextFeatureGenerator._embedding_look_up.update(loaded)

        # Transform with data that has a mix of seen and unseen text
        X_new = pd.DataFrame({"text": ["hello world", "brand new text", "foo bar baz", "never seen before"]})
        X_out = gen._transform(X_new)

        assert X_out.shape == (4, self.EMB_DIM)
        assert not X_out.isnull().any().any()

    def test_full_pipeline_cache_roundtrip_with_e5_model(self, tmp_path, monkeypatch):
        """End-to-end via TabArenaModelAgnosticPreprocessing with intfloat/e5-small-v2."""
        from sentence_transformers import SentenceTransformer

        # Monkey patch Small model for tests
        fast_model = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2", truncate_dim=4)
        monkeypatch.setattr(TabArenaDefaultTextEncoder, "get_default_encoder", lambda: fast_model)

        X = pd.DataFrame(
            {
                "description": [
                    f"This is a detailed text description for sample number {i} with unique content" for i in range(50)
                ]
            }
        )

        preprocessing = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=True,
            enable_new_datetime_features=False,
            enable_text_special_features=False,
            enable_statistical_text_features=False,
            enable_text_ngram_features=False,
            enable_datetime_features=False,
            verbosity=0,
        )

        # 1. fit_transform through the full pipeline
        X_out_fit = preprocessing.fit_transform(X=X)
        assert not X_out_fit.empty

        cache = dict(SemanticTextFeatureGenerator._embedding_look_up)
        assert len(cache) > 0

        # 2. Save cache to parquet
        cache_path = tmp_path / "pipeline_cache.parquet"
        SemanticTextFeatureGenerator.save_embedding_cache(cache=cache, path=cache_path)

        # 3. Clear and reload cache from disk
        SemanticTextFeatureGenerator._embedding_look_up.clear()
        loaded = SemanticTextFeatureGenerator.load_embedding_cache(cache_path)
        SemanticTextFeatureGenerator._embedding_look_up.update(loaded)
        assert set(loaded.keys()) == set(cache.keys())

        # 4. Transform same data with the loaded cache
        preprocessing = TabArenaModelAgnosticPreprocessing(
            enable_sematic_text_features=True,
            enable_new_datetime_features=False,
            enable_text_special_features=False,
            enable_statistical_text_features=False,
            enable_text_ngram_features=False,
            enable_datetime_features=False,
            verbosity=0,
        )
        X_out_cached = preprocessing.fit_transform(X)

        # 5. Output must be identical
        pd.testing.assert_frame_equal(X_out_fit, X_out_cached)


# ===========================================================================
# TextEmbeddingDimensionalityReductionFeatureGenerator
# ===========================================================================


class TestTextEmbeddingDRConstants:
    def test_max_components_per_batch(self):
        assert TextEmbeddingDimensionalityReductionFeatureGenerator._MAX_COMPONENTS_PER_BATCH == 30

    def test_explained_variance_threshold(self):
        assert TextEmbeddingDimensionalityReductionFeatureGenerator._EXPLAINED_VARIANCE_THRESHOLD == 0.99

    def test_default_max_features_per_group_is_inf(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        assert gen.max_features_per_group == float("inf")

    def test_custom_max_features_per_group(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator(max_features_per_group=50)
        assert gen.max_features_per_group == 50


class TestTextEmbeddingDRParseSourceColumn:
    def test_dot_separator(self):
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column("my_col.emb_0")
        assert result == "my_col"

    def test_multiple_dots_uses_first(self):
        # Only the first "." separator is used.
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column(
            "col.semantic_embedding.extra"
        )
        assert result == "col"

    def test_no_dot_returns_whole_name(self):
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column("emb_0")
        assert result == "emb_0"

    def test_empty_prefix(self):
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column(".emb_0")
        assert result == ""

    def test_statistical_suffix_convention(self):
        # StatisticalTextFeatureGenerator produces "text.42"
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column("text.42")
        assert result == "text"

    def test_semantic_suffix_convention(self):
        # SemanticTextFeatureGenerator produces "description.semantic_embedding_5"
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column(
            "description.semantic_embedding_5"
        )
        assert result == "description"

    def test_text_special_char_count_convention(self):
        # TextSpecialFeatureGenerator produces "col.char_count"
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column("col.char_count")
        assert result == "col"

    def test_dr_output_convention(self):
        # DR output is "{col}.dr{b}_{i}"
        result = TextEmbeddingDimensionalityReductionFeatureGenerator._parse_source_column("mytext.dr0_3")
        assert result == "mytext"


class TestTextEmbeddingDRMakeBatchPlan:
    def _gen(self, max_n=float("inf")):
        return TextEmbeddingDimensionalityReductionFeatureGenerator(max_features_per_group=max_n)

    def test_single_source_no_split(self):
        gen = self._gen()
        features = ["src.emb_0", "src.emb_1", "src.emb_2"]
        plan = gen._make_batch_plan(features)
        assert len(plan) == 1
        assert plan[0][0] == "src"
        assert plan[0][1] == 0
        assert plan[0][2] == features

    def test_two_sources_two_batches(self):
        gen = self._gen()
        features = ["a.emb_0", "a.emb_1", "b.emb_0"]
        plan = gen._make_batch_plan(features)
        assert len(plan) == 2
        sources = [p[0] for p in plan]
        assert "a" in sources
        assert "b" in sources

    def test_max_features_per_group_splits(self):
        gen = self._gen(max_n=2)
        features = ["src.emb_0", "src.emb_1", "src.emb_2"]
        plan = gen._make_batch_plan(features)
        # 3 features with max 2 → 2 sub-batches
        assert len(plan) == 2
        assert plan[0][1] == 0
        assert plan[1][1] == 1

    def test_max_features_per_group_no_split_when_exact(self):
        gen = self._gen(max_n=3)
        features = ["src.emb_0", "src.emb_1", "src.emb_2"]
        plan = gen._make_batch_plan(features)
        assert len(plan) == 1

    def test_no_dot_in_feature_name_each_is_its_own_group(self):
        gen = self._gen()
        features = ["emb_0", "emb_1"]
        plan = gen._make_batch_plan(features)
        # Each feature is its own source column (no dot separator)
        assert len(plan) == 2

    def test_plan_covers_all_features(self):
        gen = self._gen(max_n=2)
        features = ["a.0", "a.1", "a.2", "b.0", "b.1"]
        plan = gen._make_batch_plan(features)
        covered = [f for _, _, feats in plan for f in feats]
        assert sorted(covered) == sorted(features)

    def test_sub_batch_indices_are_sequential_per_source(self):
        gen = self._gen(max_n=2)
        features = [f"src.{i}" for i in range(5)]
        plan = gen._make_batch_plan(features)
        src_plan = [(p[1], p[2]) for p in plan if p[0] == "src"]
        indices = [p[0] for p in src_plan]
        assert indices == list(range(len(indices)))


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
        k = TextEmbeddingDimensionalityReductionFeatureGenerator._num_components_for_variance(np.array([0.001]), 0.99)
        assert k >= 1

    def test_standard_scale_fit_zero_mean(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10.0, 10.0, 10.0]})
        X_scaled, _means, _stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
        # Scaled data should have approx zero mean
        assert abs(X_scaled["a"].mean()) < 1e-10

    def test_standard_scale_fit_unit_std(self):
        X = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0, 5.0]})
        X_scaled, _means, _stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X)
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
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X.copy())
        X_transform = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_transform(X, means, stds)
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
    def _make_embedding_df(self, n_rows=30, n_cols=50, n_sources=1) -> pd.DataFrame:
        """Create a DataFrame of embedding features with proper source-column naming.

        Columns are named ``src{s}.emb{i}`` (dot separator) so that
        ``_parse_source_column`` returns the source group ``src{s}``.
        """
        rng = np.random.default_rng(1)
        cols_per_source = n_cols // n_sources
        cols = [f"src{s}.emb{i}" for s in range(n_sources) for i in range(cols_per_source)]
        data = rng.standard_normal((n_rows, len(cols)))
        return pd.DataFrame(data, columns=cols)

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

    def test_fit_preprocess_and_transform_output_cols_named_with_source_and_pca(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()  # single source "src0"
        y = self._make_target()
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        # With single source "src0", output cols should be "src0.dr0_{i}"
        import re as _re

        assert all(_re.match(r"^src0\.dr\d+_\d+$", c) for c in X_out.columns)

    def test_fit_preprocess_and_transform_max_components_per_batch(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=100, n_cols=50)
        y = self._make_target(n_rows=100)
        X_out = gen._fit_preprocess_and_transform(X=X, y=y)
        # At most _MAX_COMPONENTS_PER_BATCH components per batch (single source → single batch)
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

    def test_fit_then_transform_works(self):
        # The sorted_feature_names_ bug is now fixed; transform must succeed.
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        X_out = gen._transform(X.copy())
        assert isinstance(X_out, pd.DataFrame)
        assert list(X_out.columns) == gen.feature_names_out_

    def test_fit_then_transform_preserves_row_count(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=30)
        y = self._make_target(n_rows=30)
        gen._fit_transform(X=X.copy(), y=y)
        X_test = self._make_embedding_df(n_rows=10)
        X_out = gen._transform(X_test)
        assert len(X_out) == 10

    def test_transform_raises_on_column_mismatch(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df()
        y = self._make_target()
        gen._fit_transform(X=X.copy(), y=y)
        X_wrong = self._make_embedding_df(n_cols=10)
        with pytest.raises(ValueError):
            gen._transform(X_wrong)

    def test_multiple_sources_produce_separate_pca_batches(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator()
        X = self._make_embedding_df(n_rows=40, n_cols=20, n_sources=2)
        y = self._make_target(n_rows=40)
        gen._fit_transform(X=X.copy(), y=y)
        # Two source columns → two PCA batches
        assert len(gen._batch_pcas_) == 2

    def test_max_features_per_group_splits_into_sub_batches(self):
        gen = TextEmbeddingDimensionalityReductionFeatureGenerator(max_features_per_group=5)
        # 10 features from one source → 2 sub-batches
        rng = np.random.default_rng(42)
        cols = [f"src0.emb{i}" for i in range(10)]
        X = pd.DataFrame(rng.standard_normal((30, 10)), columns=cols)
        y = self._make_target(n_rows=30)
        gen._fit_transform(X=X.copy(), y=y)
        assert len(gen._batch_pcas_) == 2

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
        X_scaled, means, stds = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_fit(X.copy())
        # Re-apply transform to the original (un-scaled) X
        X_t = TextEmbeddingDimensionalityReductionFeatureGenerator._standard_scale_transform(X, means, stds)
        pd.testing.assert_frame_equal(X_scaled, X_t)


# ===========================================================================
# GroupAggregationFeatureGenerator
# ===========================================================================

from tabarena.benchmark.preprocessing.group_feature_generators import (
    GROUP_INDEX_FEATURES,
    GroupAggregationFeatureGenerator,
)


def _make_grouped_df(
    n_groups: int = 5,
    rows_per_group: int = 4,
    rng: np.random.Generator | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Build a simple grouped DataFrame with numeric + categorical columns."""
    rng = rng or np.random.default_rng(42)
    n_rows = n_groups * rows_per_group
    groups = np.repeat(np.arange(n_groups), rows_per_group)
    X = pd.DataFrame(
        {
            "gid": groups,
            "num_a": rng.standard_normal(n_rows) + groups * 10,
            "num_b": rng.uniform(0, 1, n_rows),
            "cat_c": rng.choice(["x", "y", "z"], n_rows),
        }
    )
    y = pd.Series(rng.standard_normal(n_rows), name="target")
    return X, y


class TestGroupAggregationFeatureGenerator:
    """Tests for the variance-based group aggregation feature generator."""

    # ------------------------------------------------------------------
    # Basic fit_transform
    # ------------------------------------------------------------------

    def test_fit_transform_returns_dataframe_and_metadata(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid")
        X_out, meta = gen._fit_transform(X.copy(), y)
        assert isinstance(X_out, pd.DataFrame)
        assert GROUP_INDEX_FEATURES in meta

    def test_group_col_dropped_from_output(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid")
        X_out, _ = gen._fit_transform(X.copy(), y)
        assert "gid" not in X_out.columns

    def test_row_count_preserved(self):
        X, y = _make_grouped_df(n_groups=3, rows_per_group=7)
        gen = GroupAggregationFeatureGenerator(group_col="gid")
        X_out, _ = gen._fit_transform(X.copy(), y)
        assert len(X_out) == 21

    def test_original_feature_columns_preserved(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid")
        X_out, _ = gen._fit_transform(X.copy(), y)
        assert "num_a" in X_out.columns
        assert "num_b" in X_out.columns
        assert "cat_c" in X_out.columns

    # ------------------------------------------------------------------
    # Aggregation columns
    # ------------------------------------------------------------------

    def test_numeric_agg_features_created(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=100)
        X_out, _ = gen._fit_transform(X.copy(), y)
        for agg in ("mean", "std", "min", "max", "last"):
            assert f"num_a_{agg}" in X_out.columns or f"num_b_{agg}" in X_out.columns

    def test_categorical_agg_features_created(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=100)
        X_out, _ = gen._fit_transform(X.copy(), y)
        for agg in ("count", "last", "nunique"):
            assert f"cat_c_{agg}" in X_out.columns

    def test_all_possible_aggs_generated_when_budget_large(self):
        """With a large budget, every (column, agg) pair is selected."""
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=200)
        X_out, _ = gen._fit_transform(X.copy(), y)
        expected_num = {f"{c}_{a}" for c in ("num_a", "num_b") for a in ("mean", "std", "min", "max", "last")}
        expected_cat = {f"cat_c_{a}" for a in ("count", "last", "nunique")}
        expected = expected_num | expected_cat
        agg_cols = set(X_out.columns) - {"num_a", "num_b", "cat_c"}
        assert expected == agg_cols

    # ------------------------------------------------------------------
    # Variance-based selection
    # ------------------------------------------------------------------

    def test_n_top_features_limits_selection(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=3)
        _X_out, meta = gen._fit_transform(X.copy(), y)
        assert len(meta[GROUP_INDEX_FEATURES]) == 3

    def test_highest_variance_feature_selected(self):
        """A column engineered to have much higher variance should be selected first."""
        rng = np.random.default_rng(99)
        n_groups, rpg = 10, 5
        n_rows = n_groups * rpg
        groups = np.repeat(np.arange(n_groups), rpg)
        X = pd.DataFrame(
            {
                "gid": groups,
                # High variance: group means spread from 0 to 9000
                "high_var": rng.standard_normal(n_rows) + groups * 1000,
                # Low variance: nearly constant
                "low_var": rng.standard_normal(n_rows) * 0.001,
            }
        )
        y = pd.Series(rng.standard_normal(n_rows))
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=1)
        gen._fit_transform(X.copy(), y)
        selected = gen._selected_features[0]
        assert selected.startswith("high_var"), f"Expected high_var_*, got {selected}"

    def test_selected_features_ordered_by_variance_descending(self):
        rng = np.random.default_rng(7)
        n_groups, rpg = 8, 5
        n_rows = n_groups * rpg
        groups = np.repeat(np.arange(n_groups), rpg)
        X = pd.DataFrame(
            {
                "gid": groups,
                "a": rng.standard_normal(n_rows) + groups * 100,
                "b": rng.standard_normal(n_rows) + groups * 10,
                "c": rng.standard_normal(n_rows) + groups * 1,
            }
        )
        y = pd.Series(rng.standard_normal(n_rows))
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=3)
        gen._fit_transform(X.copy(), y)
        # The top 3 should all come from column 'a' (highest inter-group spread)
        assert all(f.startswith("a_") for f in gen._selected_features)

    # ------------------------------------------------------------------
    # generate_index_features=False
    # ------------------------------------------------------------------

    def test_generate_features_false_drops_group_col_only(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", generate_index_features=False)
        X_out, meta = gen._fit_transform(X.copy(), y)
        assert "gid" not in X_out.columns
        assert set(X_out.columns) == {"num_a", "num_b", "cat_c"}
        assert meta == {}

    # ------------------------------------------------------------------
    # Transform (test-time)
    # ------------------------------------------------------------------

    def test_transform_produces_same_columns_as_fit_transform(self):
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        X_fit, _ = gen._fit_transform(X.copy(), y)
        X_trans = gen._transform(X.copy())
        assert list(X_fit.columns) == list(X_trans.columns)

    def test_transform_on_new_data(self):
        X_train, y = _make_grouped_df(n_groups=5, rows_per_group=4)
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        gen._fit_transform(X_train.copy(), y)
        # Build new data with same groups but different values
        rng = np.random.default_rng(123)
        X_test = pd.DataFrame(
            {
                "gid": [0, 1, 2],
                "num_a": rng.standard_normal(3),
                "num_b": rng.uniform(0, 1, 3),
                "cat_c": ["x", "y", "z"],
            }
        )
        X_out = gen._transform(X_test)
        assert len(X_out) == 3
        assert "gid" not in X_out.columns

    def test_transform_aggregation_values_are_per_group(self):
        """Rows in the same group should have identical aggregation feature values."""
        X, y = _make_grouped_df(n_groups=3, rows_per_group=4)
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        X_out, _ = gen._fit_transform(X.copy(), y)
        agg_cols = [c for c in X_out.columns if c not in ("num_a", "num_b", "cat_c")]
        for col in agg_cols:
            for gid in range(3):
                group_rows = X_out.iloc[gid * 4 : (gid + 1) * 4]
                vals = group_rows[col].dropna().unique()
                assert len(vals) <= 1, f"Non-constant agg value for group {gid}, col {col}"

    # ------------------------------------------------------------------
    # Composite group key
    # ------------------------------------------------------------------

    def test_composite_group_key(self):
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "g1": ["a", "a", "b", "b", "a", "b"],
                "g2": [1, 1, 1, 1, 2, 2],
                "val": rng.standard_normal(6),
            }
        )
        y = pd.Series(rng.standard_normal(6))
        gen = GroupAggregationFeatureGenerator(group_col=["g1", "g2"], n_top_features=5)
        X_out, _ = gen._fit_transform(X.copy(), y)
        assert "g1" not in X_out.columns
        assert "g2" not in X_out.columns
        assert "val" in X_out.columns

    # ------------------------------------------------------------------
    # Time-based sorting
    # ------------------------------------------------------------------

    def test_group_time_on_affects_last(self):
        """With group_time_on, 'last' should return the value from the latest timestamp."""
        X = pd.DataFrame(
            {
                "gid": [0, 0, 0, 1, 1, 1],
                "ts": pd.to_datetime(
                    ["2020-01-03", "2020-01-01", "2020-01-02", "2020-02-01", "2020-02-03", "2020-02-02"]
                ),
                "val": [30.0, 10.0, 20.0, 100.0, 300.0, 200.0],
            }
        )
        y = pd.Series([1.0] * 6)
        gen = GroupAggregationFeatureGenerator(group_col="gid", group_time_on="ts", n_top_features=100)
        X_out, _ = gen._fit_transform(X.copy(), y)
        # Group 0: sorted by ts → [10, 20, 30], last = 30
        # Group 1: sorted by ts → [100, 200, 300], last = 300
        last_col = "val_last"
        assert last_col in X_out.columns
        group0_last = X_out[last_col].iloc[0]
        group1_last = X_out[last_col].iloc[3]
        assert group0_last == pytest.approx(30.0)
        assert group1_last == pytest.approx(300.0)

    # ------------------------------------------------------------------
    # Determinism
    # ------------------------------------------------------------------

    def test_deterministic_selection(self):
        X, y = _make_grouped_df()
        gen1 = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        gen1._fit_transform(X.copy(), y)
        gen2 = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        gen2._fit_transform(X.copy(), y)
        assert gen1._selected_features == gen2._selected_features

    # ------------------------------------------------------------------
    # Edge cases
    # ------------------------------------------------------------------

    def test_n_top_features_exceeds_total_selects_all(self):
        """When budget > total features, all features are selected."""
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=500)
        _, meta = gen._fit_transform(X.copy(), y)
        # 2 numeric cols × 5 aggs + 1 cat col × 3 aggs = 13
        assert len(meta[GROUP_INDEX_FEATURES]) == 13

    def test_single_group(self):
        """All rows in one group: agg features are constant (zero variance)."""
        rng = np.random.default_rng(0)
        X = pd.DataFrame(
            {
                "gid": [0, 0, 0, 0],
                "val": rng.standard_normal(4),
            }
        )
        y = pd.Series(rng.standard_normal(4))
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=5)
        X_out, _ = gen._fit_transform(X.copy(), y)
        assert "gid" not in X_out.columns
        assert len(X_out) == 4

    def test_generate_features_false_transform(self):
        """Transform after fit with generate_features=False just drops group col."""
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", generate_index_features=False)
        gen._fit_transform(X.copy(), y)
        X_out = gen._transform(X.copy())
        assert "gid" not in X_out.columns
        assert set(X_out.columns) == {"num_a", "num_b", "cat_c"}

    def test_agg_maps_built_correctly(self):
        """Internal _num_agg_map and _cat_agg_map should only contain selected aggs."""
        X, y = _make_grouped_df()
        gen = GroupAggregationFeatureGenerator(group_col="gid", n_top_features=100)
        gen._fit_transform(X.copy(), y)
        # All numeric aggs for num_a and num_b should be present
        for col in ("num_a", "num_b"):
            assert col in gen._num_agg_map
            assert set(gen._num_agg_map[col]) == {"mean", "std", "min", "max", "last"}
        # All categorical aggs for cat_c
        assert "cat_c" in gen._cat_agg_map
        assert set(gen._cat_agg_map["cat_c"]) == {"count", "last", "nunique"}
