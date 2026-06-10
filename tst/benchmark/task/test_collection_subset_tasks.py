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
        tabarena_task_name="test_task",
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

    def test_tabarena_v0pt1_lite_is_one_split_per_dataset(self):
        full = TaskMetadataCollection.from_preset("TabArena-v0.1")
        lite = TaskMetadataCollection.from_preset("TabArena-v0.1-lite")
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
