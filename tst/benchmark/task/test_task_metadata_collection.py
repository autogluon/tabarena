"""Tests for `TaskMetadataCollection` — a list[TabArenaTaskMetadata] plus derived views."""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import (
    SplitMetadata,
    TabArenaTaskMetadata,
    TaskMetadataCollection,
)

# ---------------------------------------------------------------------------
# Helpers (mirror tst/benchmark/task/test_metadata_bundle.py)
# ---------------------------------------------------------------------------


def _split_meta(repeat: int = 0, fold: int = 0, num_instances_train: int = 80) -> SplitMetadata:
    return SplitMetadata(
        repeat=repeat,
        fold=fold,
        num_instances_train=num_instances_train,
        num_instances_test=20,
        num_instance_groups_train=num_instances_train,
        num_instance_groups_test=20,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=5,
        num_features_test=5,
    )


def _task_meta(
    *,
    dataset_name: str = "test_ds",
    problem_type: str = "binary",
    task_id_str: str | None = "360",
    num_features: int = 5,
    num_classes: int = 2,
    splits: list[SplitMetadata] | None = None,
) -> TabArenaTaskMetadata:
    if splits is None:
        splits = [_split_meta(repeat=0, fold=0)]
    splits_metadata = {sm.split_index: sm for sm in splits}
    return TabArenaTaskMetadata(
        dataset_name=dataset_name,
        problem_type=problem_type,
        is_classification=(problem_type != "regression"),
        target_name="target",
        eval_metric="roc_auc",
        splits_metadata=splits_metadata,
        split_time_horizon=None,
        split_time_horizon_unit=None,
        stratify_on=None,
        time_on=None,
        group_on=None,
        group_time_on=None,
        group_labels=None,
        multiclass_min_n_classes_over_splits=num_classes,
        multiclass_max_n_classes_over_splits=num_classes,
        class_consistency_over_splits=True,
        num_instances=100,
        num_features=num_features,
        num_classes=num_classes,
        num_instance_groups=100,
        tabarena_task_name=dataset_name,
        task_id_str=task_id_str,
    )


def _unrolled(ttm: TabArenaTaskMetadata) -> list[TabArenaTaskMetadata]:
    """Mirror the bundle: one entry per split."""
    return ttm.unroll_splits()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestListLike:
    def test_len_iter_getitem_tasks(self):
        tasks = [_task_meta(dataset_name="a"), _task_meta(dataset_name="b")]
        c = TaskMetadataCollection(tasks)
        assert len(c) == 2
        assert c[0].dataset_name == "a"
        assert [t.dataset_name for t in c] == ["a", "b"]
        assert c.tasks == tasks

    def test_empty(self):
        c = TaskMetadataCollection([])
        assert len(c) == 0
        assert c.dataset_names() == []
        assert c.dataset_fold_repeats() == []
        assert c.per_dataset_frame().empty
        assert c.dataset_to_tid() == {}


class TestNativeViews:
    def test_dataset_names_dedup_and_order(self):
        # 3 unrolled splits of "a" then 1 of "b" -> ["a", "b"]
        tasks = _unrolled(_task_meta(dataset_name="a", splits=[_split_meta(fold=f) for f in range(3)]))
        tasks += _unrolled(_task_meta(dataset_name="b"))
        c = TaskMetadataCollection(tasks)
        assert c.dataset_names() == ["a", "b"]

    def test_dataset_fold_repeats_from_splits(self):
        splits = [_split_meta(repeat=r, fold=f) for r in range(2) for f in range(3)]  # 2x3
        c = TaskMetadataCollection(_unrolled(_task_meta(dataset_name="a", splits=splits)))
        dfr = c.dataset_fold_repeats()
        assert len(dfr) == 6
        assert set(dfr) == {("a", f, r) for r in range(2) for f in range(3)}

    def test_task_grid_exposes_warehouse_predicate_columns(self):
        # task_grid carries the warehouse columns (None for tasks that don't set them, e.g.
        # TabArena v0.1) so arena-specific predicates (BeyondArena) can read them.
        c = TaskMetadataCollection(_unrolled(_task_meta(dataset_name="a")))
        grid = c.task_grid()
        for col in ("task_type", "num_cols_after_preprocessing", "num_text_cols", "num_high_cardinality_cats"):
            assert col in grid.columns

    def test_per_dataset_frame_one_row_per_dataset_native_columns(self):
        tasks = _unrolled(_task_meta(dataset_name="a", num_features=7, splits=[_split_meta(fold=f) for f in range(3)]))
        tasks += _unrolled(_task_meta(dataset_name="b", problem_type="regression", num_features=4))
        frame = TaskMetadataCollection(tasks).per_dataset_frame()
        assert len(frame) == 2  # collapsed across the 3 splits of "a"
        # native column names + a `dataset` key
        for col in ("dataset", "problem_type", "num_features", "num_classes"):
            assert col in frame.columns
        by_ds = frame.set_index("dataset")
        assert by_ds.loc["a", "num_features"] == 7
        assert by_ds.loc["b", "problem_type"] == "regression"


def _legacy_row(
    *,
    dataset: str = "ds",
    tid: int = 363612,
    problem_type: str = "binary",
    n_features: int = 7,
    n_classes: int = 2,
    n_folds: int = 3,
    n_repeats: int = 1,
    n_instances: int = 100,
) -> dict:
    """One row in the `load_task_metadata` (legacy) format."""
    return {
        "dataset": dataset,
        "name": dataset,
        "tid": tid,
        "problem_type": problem_type,
        "n_features": n_features,
        "n_classes": n_classes,
        "n_folds": n_folds,
        "n_repeats": n_repeats,
        "NumberOfInstances": n_instances,
        "n_samples_train_per_fold": int(n_instances * 2 / 3),
        "n_samples_test_per_fold": int(n_instances / 3),
        "target_feature": "target",
    }


class TestFromLegacyDf:
    def test_basic_reconstruction(self):
        df = pd.DataFrame([_legacy_row(dataset="a", n_features=7, n_classes=2)])
        c = TaskMetadataCollection.from_legacy_df(df)
        assert isinstance(c, TaskMetadataCollection)
        t = c.tasks[0]
        assert t.dataset_name == "a" and t.tabarena_task_name == "a"
        assert t.problem_type == "binary" and t.is_classification is True
        assert t.num_features == 7 and t.num_classes == 2
        assert t.target_name == "target"
        assert t.task_id_str == "363612"
        assert t.eval_metric == "roc_auc"  # derived from problem_type

    def test_splits_expanded(self):
        df = pd.DataFrame([_legacy_row(n_folds=3, n_repeats=2)])
        c = TaskMetadataCollection.from_legacy_df(df)
        assert len(c.dataset_fold_repeats()) == 6  # 3 folds x 2 repeats

    def test_eval_metric_derivation(self):
        df = pd.DataFrame([_legacy_row(problem_type="regression", n_classes=0)])
        t = TaskMetadataCollection.from_legacy_df(df).tasks[0]
        assert t.eval_metric == "rmse"
        assert t.is_classification is False
        assert t.num_classes == -1  # regression normalized to the schema's -1 (from legacy 0)

    def test_lossy_fields_are_none(self):
        t = TaskMetadataCollection.from_legacy_df(pd.DataFrame([_legacy_row()])).tasks[0]
        assert t.has_text is None
        assert t.domain is None
        assert t.group_on is None
        assert t.multiclass_max_n_classes_over_splits is None

    def test_missing_required_column_raises(self):
        df = pd.DataFrame([_legacy_row()]).drop(columns=["n_folds"])
        with pytest.raises(ValueError, match="missing required columns"):
            TaskMetadataCollection.from_legacy_df(df)

    def test_roundtrip_preserves_kept_fields(self):
        df = pd.DataFrame([_legacy_row(dataset="a", n_features=7, n_folds=3, n_repeats=2)])
        legacy_again = TaskMetadataCollection.from_legacy_df(df).to_legacy_df()
        row = legacy_again.iloc[0]
        assert row["dataset"] == "a"
        assert row["n_features"] == 7
        assert row["n_folds"] == 3
        assert row["n_repeats"] == 2
        assert row["tid"] == 363612

    def test_real_legacy_df_matches_native_preset(self):
        """The real legacy df rebuilds to the same datasets/feature counts as the native preset."""
        from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

        legacy_collection = TaskMetadataCollection.from_legacy_df(load_task_metadata(paper=True))
        native = TaskMetadataCollection.from_preset("TabArena-v0.1")
        assert set(legacy_collection.dataset_names()) == set(native.dataset_names())  # 51 datasets
        legacy_nf = legacy_collection.per_dataset_frame().set_index("dataset")["num_features"]
        native_nf = native.per_dataset_frame().set_index("dataset")["num_features"]
        assert (legacy_nf.sort_index() == native_nf.sort_index()).all()


class TestLegacyBoundary:
    def test_dataset_to_tid_parses_task_id_str(self):
        tasks = [
            _task_meta(dataset_name="a", task_id_str="363612"),
            _task_meta(dataset_name="b", task_id_str="UserTask|9900335484|ds/uuid"),
        ]
        assert TaskMetadataCollection(tasks).dataset_to_tid() == {"a": 363612, "b": 9900335484}

    def test_to_legacy_df_delegates(self):
        c = TaskMetadataCollection(_unrolled(_task_meta(splits=[_split_meta(fold=f) for f in range(3)])))
        legacy = c.to_legacy_df()
        assert isinstance(legacy, pd.DataFrame)
        assert len(legacy) == 1  # one row per dataset
        for col in ("dataset", "tid", "n_folds", "n_features"):
            assert col in legacy.columns
        assert legacy.iloc[0]["n_folds"] == 3


class TestSubset:
    def test_keeps_requested_splits_and_drops_unrequested_tasks(self):
        a = _task_meta(dataset_name="a", splits=[_split_meta(repeat=r, fold=f) for r in range(2) for f in range(2)])
        b = _task_meta(dataset_name="b", splits=[_split_meta(fold=0)])
        c = TaskMetadataCollection([a, b])
        # Keep two of a's four splits; b is not requested, so it drops out entirely.
        sub = c.subset([("a", 0, 0), ("a", 1, 1)])
        assert sub.dataset_names() == ["a"]
        assert set(sub.dataset_fold_repeats()) == {("a", 0, 0), ("a", 1, 1)}

    def test_invalid_triplet_raises(self):
        c = TaskMetadataCollection([_task_meta(dataset_name="a", splits=[_split_meta(fold=0)])])
        with pytest.raises(ValueError, match="not splits of this collection"):
            c.subset([("a", 1, 0)])
