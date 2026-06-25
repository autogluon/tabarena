"""Tests for `TaskMetadataCollection.from_source` / `from_preset` / `subset_tasks`.

Ported from the pre-refactor `test_metadata_bundle.py` (which tested
`TabArenaMetadataBundle.load_task_metadata`): construction + declarative filtering
now live directly on `TaskMetadataCollection`.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import (
    SplitMetadata,
    TabArenaTaskMetadata,
    TaskMetadataCollection,
    TaskSubset,
)
from tabarena.benchmark.task.metadata.sources import load_tabarena_v0_1_task_metadata

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _split_meta(repeat: int = 0, fold: int = 0, n_train: int = 80) -> SplitMetadata:
    return SplitMetadata(
        repeat=repeat,
        fold=fold,
        num_instances_train=n_train,
        num_instances_test=20,
        num_instance_groups_train=n_train,
        num_instance_groups_test=20,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=5,
        num_features_test=5,
    )


def _task_meta(
    problem_type: str = "binary",
    task_id_str: str | None = "360",
    n_splits: int = 1,
    dataset_name: str = "test_ds",
    n_train: int = 80,
    **extra_fields,
) -> TabArenaTaskMetadata:
    splits: dict = {}
    for i in range(n_splits):
        sm = _split_meta(repeat=0, fold=i, n_train=n_train)
        splits[sm.split_index] = sm
    return TabArenaTaskMetadata(
        dataset_name=dataset_name,
        problem_type=problem_type,
        is_classification=(problem_type != "regression"),
        target_name="target",
        eval_metric="roc_auc",
        splits_metadata=splits,
        split_time_horizon=None,
        split_time_horizon_unit=None,
        stratify_on=None,
        time_on=None,
        group_on=None,
        group_time_on=None,
        group_labels=None,
        multiclass_min_n_classes_over_splits=2,
        multiclass_max_n_classes_over_splits=2,
        class_consistency_over_splits=True,
        num_instances=100,
        num_features=5,
        num_classes=2,
        num_instance_groups=100,
        tabarena_task_name=dataset_name,
        task_id_str=task_id_str,
        **extra_fields,
    )


def _collection(task_metadata) -> TaskMetadataCollection:
    return TaskMetadataCollection.from_source(task_metadata)


# ---------------------------------------------------------------------------
# from_source — list / DataFrame inputs
# ---------------------------------------------------------------------------


class TestFromSource:
    def test_list_input_passthrough(self):
        result = _collection([_task_meta()])
        assert len(result) == 1
        assert result[0].dataset_name == "test_ds"

    def test_multi_split_task_unrolled(self):
        result = _collection([_task_meta(n_splits=3)])
        assert len(result) == 3

    def test_each_task_has_exactly_one_split(self):
        result = _collection([_task_meta(n_splits=4)])
        for item in result:
            assert item.n_splits == 1

    def test_missing_task_id_str_raises(self):
        with pytest.raises(ValueError, match="task_id_str"):
            _collection([_task_meta(task_id_str=None)])

    def test_multiple_tasks_combined(self):
        meta = [
            _task_meta(dataset_name="a", n_splits=2),
            _task_meta(dataset_name="b", n_splits=3),
        ]
        assert len(_collection(meta)) == 5

    def test_empty_task_list(self):
        assert len(_collection([])) == 0

    def test_dataframe_input_parsed(self):
        meta_obj = _task_meta(n_splits=1)
        df = meta_obj.to_dataframe()
        result = _collection(df)
        assert len(result) == 1
        assert result[0].dataset_name == meta_obj.dataset_name

    def test_source_is_retained_and_materialize_is_noop(self):
        result = _collection([_task_meta()])
        assert result.source is not None
        assert result.materialize() is result

    def test_to_dataframe_round_trips(self):
        original = _collection([_task_meta(dataset_name="a", n_splits=2), _task_meta(dataset_name="b")])
        reloaded = _collection(original.to_dataframe())
        assert reloaded == original


class TestFromPreset:
    def test_unknown_preset_raises(self):
        with pytest.raises(ValueError, match="Unknown preset"):
            TaskMetadataCollection.from_preset("not-a-suite")

    def test_lite_split_filter_is_one_split_per_dataset(self):
        full = TaskMetadataCollection.from_preset("TabArena-v0.1")
        lite = full.subset_tasks(split_indices="lite")
        assert len(lite) == len(lite.dataset_names())
        assert set(lite.dataset_names()) == set(full.dataset_names())
        assert all(t.split_index == "r0f0" for t in lite)


# ---------------------------------------------------------------------------
# subset_tasks — declarative filters
# ---------------------------------------------------------------------------


class TestSubsetTasks:
    def test_no_filters_returns_same_tasks(self):
        collection = _collection([_task_meta(n_splits=2)])
        assert collection.subset_tasks() == collection

    def test_problem_type_filter_excludes_non_matching(self):
        meta = [
            _task_meta(problem_type="binary", dataset_name="bin_ds"),
            _task_meta(problem_type="regression", dataset_name="reg_ds"),
        ]
        result = _collection(meta).subset_tasks(problem_types=["binary"])
        assert all(m.problem_type == "binary" for m in result)
        assert not any(m.dataset_name == "reg_ds" for m in result)

    def test_problem_type_filter_keeps_all_listed_types(self):
        meta = [
            _task_meta(problem_type="binary"),
            _task_meta(problem_type="multiclass"),
            _task_meta(problem_type="regression"),
        ]
        result = _collection(meta).subset_tasks(problem_types=["binary", "multiclass", "regression"])
        assert len(result) == 3

    def test_split_indices_none_keeps_all(self):
        result = _collection([_task_meta(n_splits=4)]).subset_tasks(split_indices=None)
        assert len(result) == 4

    def test_split_indices_lite_keeps_only_r0f0(self):
        result = _collection([_task_meta(n_splits=4)]).subset_tasks(split_indices="lite")
        assert len(result) == 1
        assert result[0].split_index == "r0f0"

    def test_split_indices_list_filters_correctly(self):
        result = _collection([_task_meta(n_splits=3)]).subset_tasks(split_indices=["r0f0", "r0f2"])
        assert len(result) == 2
        assert {m.split_index for m in result} == {"r0f0", "r0f2"}

    def test_split_indices_invalid_format_raises(self):
        with pytest.raises(ValueError, match="SplitIndex format"):
            _collection([_task_meta()]).subset_tasks(split_indices=["fold0"])

    def test_dataset_names_filter(self):
        meta = [_task_meta(dataset_name="a"), _task_meta(dataset_name="b")]
        result = _collection(meta).subset_tasks(dataset_names=["a"])
        assert [t.dataset_name for t in result] == ["a"]

    def test_dataset_names_unknown_raises(self):
        with pytest.raises(ValueError, match="not found in task metadata"):
            _collection([_task_meta(dataset_name="a")]).subset_tasks(dataset_names=["nope"])

    def test_n_train_samples_band(self):
        meta = [
            _task_meta(dataset_name="small", n_train=50),
            _task_meta(dataset_name="big", n_train=5000),
        ]
        result = _collection(meta).subset_tasks(n_train_samples=(0, 100))
        assert [t.dataset_name for t in result] == ["small"]
        # Lower bound is exclusive, upper inclusive.
        assert len(_collection(meta).subset_tasks(n_train_samples=(50, 5000))) == 1
        assert len(_collection(meta).subset_tasks(n_train_samples=(49, 5000))) == 2

    def test_dtype_filters(self):
        meta = [
            _task_meta(dataset_name="with_text", has_text=True, has_numerical=True),
            _task_meta(dataset_name="no_text", has_text=False, has_numerical=True),
        ]
        required = _collection(meta).subset_tasks(required_dtypes=["text"])
        assert [t.dataset_name for t in required] == ["with_text"]
        forbidden = _collection(meta).subset_tasks(forbidden_dtypes=["text"])
        assert [t.dataset_name for t in forbidden] == ["no_text"]

    def test_filters_compose_and_preserve_source(self):
        meta = [
            _task_meta(dataset_name="a", n_splits=3),
            _task_meta(dataset_name="b", problem_type="regression"),
        ]
        collection = _collection(meta)
        result = collection.subset_tasks(problem_types=["binary"], split_indices="lite")
        assert len(result) == 1
        assert result[0].dataset_name == "a"
        assert result.source is collection.source


# ---------------------------------------------------------------------------
# subset_tasks — named subset predicates (the arena-context shorthand)
# ---------------------------------------------------------------------------


class TestSubsetPredicateFilter:
    def _coll(self) -> TaskMetadataCollection:
        return _collection(
            [
                _task_meta(dataset_name="bin_small", problem_type="binary", n_train=500),
                _task_meta(dataset_name="reg_big", problem_type="regression", n_train=50_000),
            ],
        )

    @staticmethod
    def _custom_predicates():
        from tabarena.nips2025_utils.subset_predicate import SubsetPredicate

        return {
            "binary": SubsetPredicate(lambda df: df["problem_type"] == "binary", ("problem_type",)),
            "big": SubsetPredicate(lambda df: df["max_train_rows"] > 1_000, ("max_train_rows",)),
        }

    def test_string_atom_filters_by_predicate(self):
        result = self._coll().subset_tasks(subset="binary", predicates=self._custom_predicates())
        assert [t.dataset_name for t in result] == ["bin_small"]

    def test_negation(self):
        result = self._coll().subset_tasks(subset="!big", predicates=self._custom_predicates())
        assert [t.dataset_name for t in result] == ["bin_small"]

    def test_union_within_expression_keeps_both(self):
        result = self._coll().subset_tasks(subset="binary|big", predicates=self._custom_predicates())
        assert {t.dataset_name for t in result} == {"bin_small", "reg_big"}

    def test_list_is_anded(self):
        result = self._coll().subset_tasks(subset=["binary", "!big"], predicates=self._custom_predicates())
        assert [t.dataset_name for t in result] == ["bin_small"]

    def test_list_of_lists_unions_views(self):
        # A list of lists is OR across views: binary datasets OR big datasets keeps both,
        # whereas the flat (AND) list ["binary", "big"] keeps neither (none is both).
        coll, preds = self._coll(), self._custom_predicates()
        assert {t.dataset_name for t in coll.subset_tasks(subset=["binary", "big"], predicates=preds)} == set()
        union = coll.subset_tasks(subset=[["binary"], ["big"]], predicates=preds)
        assert {t.dataset_name for t in union} == {"bin_small", "reg_big"}

    def test_list_of_lists_ands_within_each_view(self):
        # Within a view the inner list is AND-ed, across views OR-ed: (binary AND not-big)
        # OR (big) -> bin_small (via view 1) plus reg_big (via view 2).
        result = self._coll().subset_tasks(subset=[["binary", "!big"], ["big"]], predicates=self._custom_predicates())
        assert {t.dataset_name for t in result} == {"bin_small", "reg_big"}

    def test_list_of_lists_unions_at_split_level(self):
        from tabarena.contexts.beyondarena_context import BeyondArenaContext

        coll = _collection(
            [
                _task_meta(dataset_name="ds_bin", problem_type="binary", n_splits=3),
                _task_meta(dataset_name="ds_reg", problem_type="regression", n_splits=3),
            ],
        )
        # lite -> r0f0 of every dataset; regression -> all splits of ds_reg. The union keeps
        # ds_bin's single lite split plus all three ds_reg splits.
        result = coll.subset_tasks(subset=[["lite"], ["regression"]], predicates=BeyondArenaContext.SUBSET_PREDICATES)
        assert sorted((t.dataset_name, t.split_index) for t in result) == [
            ("ds_bin", "r0f0"),
            ("ds_reg", "r0f0"),
            ("ds_reg", "r0f1"),
            ("ds_reg", "r0f2"),
        ]

    def test_invalid_name_raises(self):
        with pytest.raises(ValueError, match="Invalid subset name"):
            self._coll().subset_tasks(subset="nope", predicates=self._custom_predicates())

    def test_beyond_arena_predicates_shorthand(self):
        from tabarena.contexts.beyondarena_context import BeyondArenaContext

        result = self._coll().subset_tasks(subset="regression", predicates=BeyondArenaContext.SUBSET_PREDICATES)
        assert [t.dataset_name for t in result] == ["reg_big"]

    def test_split_level_lite_predicate_keeps_first_split(self):
        from tabarena.contexts.beyondarena_context import BeyondArenaContext

        coll = _collection([_task_meta(dataset_name="a", n_splits=3, problem_type="binary", n_train=500)])
        result = coll.subset_tasks(subset="lite", predicates=BeyondArenaContext.SUBSET_PREDICATES)
        assert len(result) == 1
        assert result[0].split_index == "r0f0"

    def test_preserves_source(self):
        collection = self._coll()
        result = collection.subset_tasks(subset="binary", predicates=self._custom_predicates())
        assert result.source is collection.source

    def test_no_provider_falls_back_to_tabarena_defaults(self):
        # A directly-built collection has no preset provider; "binary" must still resolve
        # via the TabArenaContext default predicates.
        result = self._coll().subset_tasks(subset="binary")
        assert [t.dataset_name for t in result] == ["bin_small"]


class TestPresetDefaultPredicates:
    """`from_preset` attaches the suite's default subset predicates, resolved lazily."""

    # Use the TabArena-v0.1 preset here (not BeyondArena): it loads from committed metadata with
    # no `data_foundry` dependency, so these run in CI. The BeyondArena provider is covered in
    # test_beyond_arena_collection.py via the patched-data-foundry fixture.
    def test_preset_uses_its_predicates_without_passing_them(self):
        collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
        # Resolving a named predicate without an explicit `predicates=` proves the preset default
        # provider is used.
        result = collection.subset_tasks(subset="binary")
        assert 0 < len(result.dataset_names()) < len(collection.dataset_names())

    def test_provider_is_lazy_and_survives_filtering(self):
        collection = TaskMetadataCollection.from_preset("TabArena-v0.1")
        # Set by from_preset as a thunk (not yet invoked), and propagated through filters.
        assert callable(collection._default_predicates_provider)
        filtered = collection.subset_tasks(problem_types=["binary"])
        assert filtered._default_predicates_provider is collection._default_predicates_provider

    def test_directly_built_collection_has_no_provider(self):
        collection = _collection([_task_meta()])
        assert collection._default_predicates_provider is None

    def test_prebuilt_tabarena_collection_matches_preset(self):
        from tabarena.benchmark.task.metadata import TabArenaTaskMetadataCollection

        prebuilt = TabArenaTaskMetadataCollection()
        preset = TaskMetadataCollection.from_preset("TabArena-v0.1")
        assert isinstance(prebuilt, TaskMetadataCollection)
        assert prebuilt == preset
        # Default predicates attached -> `subset=` works without `predicates=`.
        assert len(prebuilt.subset_tasks(subset="binary").dataset_names()) > 0


# ---------------------------------------------------------------------------
# to_dataframe(add_old_minimal_metadata=True) — legacy tid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("task_id_str", "expected_tid"),
    [
        ("363612", 363612),  # plain OpenML integer id (e.g. v0.1)
        ("UserTask|9900335484|ds/uuid", 9900335484),  # local UserTask id
    ],
    ids=["openml_int", "user_task"],
)
def test_add_old_minimal_metadata_tid_handles_both_id_formats(task_id_str, expected_tid):
    meta = _task_meta(task_id_str=task_id_str)
    df = meta.to_dataframe(add_old_minimal_metadata=True)
    assert df["tid"].iloc[0] == expected_tid


# ---------------------------------------------------------------------------
# TabArena v0.1 rebuild conversion (+ collection filtering on top)
# ---------------------------------------------------------------------------


def _fake_curated_metadata(rows: list[dict] | None = None) -> pd.DataFrame:
    """Build a minimal DataFrame that mimics load_curated_task_metadata()."""
    if rows is None:
        rows = [
            {
                "dataset_name": "fake_ds",
                "problem_type": "binary",
                "is_classification": True,
                "target_feature": "target",
                "task_id": "999",
                "num_instances": 100,
                "num_features": 5,
                "num_classes": 2,
                "tabarena_num_repeats": 1,
                "num_folds": 3,
            },
        ]
    return pd.DataFrame(rows)


class TestTabArenaV0pt1Conversion:
    """Tests for load_tabarena_v0_1_task_metadata (the v0.1 rebuild) and collection filtering on top."""

    def test_returns_task_metadata_list(self):
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        assert isinstance(result, list)
        assert len(result) > 0
        assert all(isinstance(r, TabArenaTaskMetadata) for r in result)

    def test_creates_one_entry_per_repeat_and_fold(self):
        """1 repeat x 3 folds = 3 entries."""
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        assert len(result) == 3

    def test_multiple_repeats(self):
        """2 repeats x 3 folds = 6 entries."""
        curated = _fake_curated_metadata(
            [
                {
                    "dataset_name": "ds_multi_repeat",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_feature": "target",
                    "task_id": "111",
                    "num_instances": 60,
                    "num_features": 3,
                    "num_classes": 2,
                    "tabarena_num_repeats": 2,
                    "num_folds": 3,
                },
            ],
        )
        assert len(load_tabarena_v0_1_task_metadata(curated)) == 6

    def test_dataset_name_propagated(self):
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        assert all(r.dataset_name == "fake_ds" for r in result)

    def test_task_id_str_propagated(self):
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        assert all(r.task_id_str == "999" for r in result)

    def test_num_features_excludes_target(self):
        """Curated num_features counts the target column; the conversion must drop it."""
        # _fake_curated_metadata default has num_features=5 -> 4 real features.
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        ttm = result[0]
        assert ttm.num_features == 4
        split = ttm.splits_metadata[ttm.split_index]
        assert split.num_features_train == 4
        assert split.num_features_test == 4

    def test_num_classes_regression_is_minus_one(self):
        """Regression tasks use the schema's -1 num_classes convention."""
        curated = _fake_curated_metadata(
            [
                {
                    "dataset_name": "reg_ds",
                    "problem_type": "regression",
                    "is_classification": False,
                    "target_feature": "t",
                    "task_id": "4",
                    "num_instances": 150,
                    "num_features": 8,
                    "num_classes": 0,  # curated leaves regression unset; conversion -> -1
                    "tabarena_num_repeats": 1,
                    "num_folds": 1,
                },
            ],
        )
        ttm = load_tabarena_v0_1_task_metadata(curated)[0]
        assert ttm.num_classes == -1
        split = ttm.splits_metadata[ttm.split_index]
        assert split.num_classes_train == -1
        assert split.num_classes_test == -1

    def test_num_classes_classification_is_int(self):
        ttm = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())[0]  # binary, num_classes=2
        assert ttm.num_classes == 2
        assert isinstance(ttm.num_classes, int)

    def test_problem_type_filter_applied_via_collection(self):
        curated = _fake_curated_metadata(
            [
                {
                    "dataset_name": "bin_ds",
                    "problem_type": "binary",
                    "is_classification": True,
                    "target_feature": "t",
                    "task_id": "1",
                    "num_instances": 50,
                    "num_features": 3,
                    "num_classes": 2,
                    "tabarena_num_repeats": 1,
                    "num_folds": 1,
                },
                {
                    "dataset_name": "reg_ds",
                    "problem_type": "regression",
                    "is_classification": False,
                    "target_feature": "t",
                    "task_id": "2",
                    "num_instances": 50,
                    "num_features": 3,
                    "num_classes": 0,
                    "tabarena_num_repeats": 1,
                    "num_folds": 1,
                },
            ],
        )
        result = _collection(load_tabarena_v0_1_task_metadata(curated)).subset_tasks(problem_types=["binary"])
        assert len(result) == 1
        assert result[0].problem_type == "binary"

    def test_split_indices_lite_filter_via_collection(self):
        """With 'lite', only r0f0 should survive out of 3 folds."""
        result = _collection(load_tabarena_v0_1_task_metadata(_fake_curated_metadata())).subset_tasks(
            split_indices="lite",
        )
        assert len(result) == 1
        assert result[0].split_index == "r0f0"

    def test_eval_metric_binary_is_roc_auc(self):
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        assert all(r.eval_metric == "roc_auc" for r in result)

    def test_eval_metric_multiclass_is_log_loss(self):
        curated = _fake_curated_metadata(
            [
                {
                    "dataset_name": "mc_ds",
                    "problem_type": "multiclass",
                    "is_classification": True,
                    "target_feature": "t",
                    "task_id": "3",
                    "num_instances": 200,
                    "num_features": 10,
                    "num_classes": 5,
                    "tabarena_num_repeats": 1,
                    "num_folds": 1,
                },
            ],
        )
        assert load_tabarena_v0_1_task_metadata(curated)[0].eval_metric == "log_loss"

    def test_eval_metric_regression_is_rmse(self):
        curated = _fake_curated_metadata(
            [
                {
                    "dataset_name": "reg_ds",
                    "problem_type": "regression",
                    "is_classification": False,
                    "target_feature": "t",
                    "task_id": "4",
                    "num_instances": 150,
                    "num_features": 8,
                    "num_classes": 0,
                    "tabarena_num_repeats": 1,
                    "num_folds": 1,
                },
            ],
        )
        assert load_tabarena_v0_1_task_metadata(curated)[0].eval_metric == "rmse"

    def test_each_result_has_one_split(self):
        result = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())
        for item in result:
            assert item.n_splits == 1

    def test_warehouse_fields_mapped_from_curated(self):
        """domain/year/source/task_type are populated; dataset-derived stats stay None."""
        curated = _fake_curated_metadata()
        curated["domain"] = "medical & healthcare"
        curated["year"] = 2014
        curated["data_source"] = "UCI"
        ttm = load_tabarena_v0_1_task_metadata(curated)[0]
        assert ttm.task_type == "random"
        assert ttm.domain == "medical & healthcare"
        assert ttm.dataset_year == "2014"  # cast to str
        assert ttm.source == "UCI"
        # No dataset is loaded for v0.1, so dataset-derived stats are unavailable.
        assert ttm.num_cols_after_preprocessing is None
        assert ttm.missing_value_fraction is None

    def test_warehouse_fields_optional_when_columns_absent(self):
        """A curated table without domain/year/source still converts (fields -> None)."""
        ttm = load_tabarena_v0_1_task_metadata(_fake_curated_metadata())[0]
        assert ttm.task_type == "random"
        assert ttm.domain is None
        assert ttm.dataset_year is None
        assert ttm.source is None


# ---------------------------------------------------------------------------
# TaskSubset — the typed single source of truth for the subset filters
# ---------------------------------------------------------------------------


class TestTaskSubset:
    def _two(self) -> TaskMetadataCollection:
        return _collection(
            [
                _task_meta(problem_type="binary", dataset_name="b"),
                _task_meta(problem_type="regression", dataset_name="r"),
            ],
        )

    def test_as_kwargs_drops_none(self):
        assert TaskSubset(subset="lite").as_kwargs() == {"subset": "lite"}
        assert TaskSubset().as_kwargs() == {}

    def test_merged_with_other_wins_per_field(self):
        base = TaskSubset(subset="lite", problem_types=["binary"])
        merged = base.merged_with(TaskSubset(subset="tiny"))
        # `other` overrides the shared field; the field it leaves unset falls back to base.
        assert merged == TaskSubset(subset="tiny", problem_types=["binary"])

    def test_from_input_none_is_empty(self):
        assert TaskSubset.from_input(None) == TaskSubset()

    def test_from_input_task_subset_is_passthrough(self):
        spec = TaskSubset(subset="lite")
        assert TaskSubset.from_input(spec) is spec

    def test_from_input_dict_constructs(self):
        assert TaskSubset.from_input({"subset": "lite"}) == TaskSubset(subset="lite")

    def test_from_input_unknown_key_raises(self):
        with pytest.raises(ValueError, match="Unknown TaskSubset field"):
            TaskSubset.from_input({"not_a_filter": 1})

    def test_from_input_bad_type_raises(self):
        with pytest.raises(TypeError):
            TaskSubset.from_input(123)

    def test_subset_tasks_accepts_spec_equivalent_to_kwargs(self):
        via_spec = self._two().subset_tasks(TaskSubset(problem_types=["binary"]))
        via_kwargs = self._two().subset_tasks(problem_types=["binary"])
        assert via_spec == via_kwargs
        assert [t.dataset_name for t in via_spec] == ["b"]

    def test_subset_tasks_loose_kwargs_override_spec(self):
        # The passed spec says regression; the loose keyword overrides it to binary.
        result = self._two().subset_tasks(TaskSubset(problem_types=["regression"]), problem_types=["binary"])
        assert [t.dataset_name for t in result] == ["b"]

    def test_subset_tasks_unknown_loose_kwarg_raises(self):
        with pytest.raises(ValueError, match="Unknown TaskSubset field"):
            self._two().subset_tasks(not_a_filter=1)
