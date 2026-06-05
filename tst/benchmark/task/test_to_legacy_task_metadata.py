"""Tests for `to_legacy_task_metadata` — converting a list of TabArenaTaskMetadata
into the legacy `task_metadata` DataFrame consumed by TabArenaContext /
ExperimentBatchRunner.
"""

from __future__ import annotations

import pandas as pd

from tabarena.benchmark.task.metadata import (
    SplitMetadata,
    TabArenaTaskMetadata,
    to_legacy_task_metadata,
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


# The columns TabArenaContext + ExperimentBatchRunner actually read from task_metadata.
# `n_samples_train_per_fold` is aliased to `max_train_rows` by the subset predicates.
_REQUIRED_LEGACY_COLUMNS = {
    "dataset",
    "tid",
    "n_folds",
    "n_repeats",
    "problem_type",
    "n_features",
    "n_classes",
    "n_samples_train_per_fold",
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestToLegacyTaskMetadata:
    def test_empty_list_returns_empty_dataframe(self):
        result = to_legacy_task_metadata([])
        assert isinstance(result, pd.DataFrame)
        assert result.empty

    def test_required_columns_present(self):
        result = to_legacy_task_metadata([_task_meta()])
        missing = _REQUIRED_LEGACY_COLUMNS - set(result.columns)
        assert not missing, f"legacy task_metadata missing required columns: {sorted(missing)}"

    def test_one_row_per_dataset_collapses_splits(self):
        # 1 repeat x 3 folds => a single dataset row (not three).
        splits = [_split_meta(repeat=0, fold=f) for f in range(3)]
        result = to_legacy_task_metadata([_task_meta(splits=splits)])
        assert len(result) == 1
        row = result.iloc[0]
        assert row["dataset"] == "test_ds"
        assert row["n_folds"] == 3
        assert row["n_repeats"] == 1

    def test_aggregate_counts_distinct_folds_and_repeats(self):
        # 2 repeats x 3 folds.
        splits = [_split_meta(repeat=r, fold=f) for r in range(2) for f in range(3)]
        result = to_legacy_task_metadata([_task_meta(splits=splits)])
        assert len(result) == 1
        assert result.iloc[0]["n_folds"] == 3
        assert result.iloc[0]["n_repeats"] == 2

    def test_train_samples_is_max_across_folds(self):
        # max_train_rows semantics => the largest per-fold train size wins.
        splits = [
            _split_meta(repeat=0, fold=0, num_instances_train=80),
            _split_meta(repeat=0, fold=1, num_instances_train=95),
        ]
        result = to_legacy_task_metadata([_task_meta(splits=splits)])
        assert result.iloc[0]["n_samples_train_per_fold"] == 95

    def test_columns_renamed_to_predicate_names(self):
        result = to_legacy_task_metadata([_task_meta(num_features=7, num_classes=3)])
        # predicate names, not the schema's num_features / num_classes
        assert "num_features" not in result.columns
        assert "num_classes" not in result.columns
        assert result.iloc[0]["n_features"] == 7
        assert result.iloc[0]["n_classes"] == 3

    def test_tid_derived_from_task_id_str(self):
        result = to_legacy_task_metadata([_task_meta(task_id_str="363612")])
        assert result.iloc[0]["tid"] == 363612

    def test_tid_derived_from_user_task_id(self):
        result = to_legacy_task_metadata([_task_meta(task_id_str="UserTask|9900335484|ds/uuid")])
        assert result.iloc[0]["tid"] == 9900335484

    def test_problem_type_preserved(self):
        result = to_legacy_task_metadata([_task_meta(problem_type="regression")])
        assert result.iloc[0]["problem_type"] == "regression"

    def test_multiple_datasets_one_row_each_no_fanout(self):
        metas = [
            _task_meta(dataset_name="a", task_id_str="1", splits=[_split_meta(fold=f) for f in range(3)]),
            _task_meta(dataset_name="b", task_id_str="2", splits=[_split_meta(fold=f) for f in range(2)]),
        ]
        result = to_legacy_task_metadata(metas)
        assert len(result) == 2
        assert set(result["dataset"]) == {"a", "b"}
        # no duplicate dataset rows (would fan out subset_results' merge)
        assert not result["dataset"].duplicated().any()
        by_ds = result.set_index("dataset")
        assert by_ds.loc["a", "n_folds"] == 3
        assert by_ds.loc["b", "n_folds"] == 2

    def test_split_grain_columns_dropped(self):
        # After collapsing to dataset grain, per-split columns are meaningless and removed.
        result = to_legacy_task_metadata([_task_meta(splits=[_split_meta(fold=f) for f in range(3)])])
        for col in ("fold", "repeat", "split_index", "num_instances_train", "num_features_train"):
            assert col not in result.columns

    def test_subset_predicate_expressions_evaluate(self):
        """The produced columns support the documented subset predicates without KeyError.

        Mirrors the real predicate logic for a couple of representative subsets
        (`tabpfn` and `small`) to prove the column contract is satisfied end-to-end.
        """
        metas = [
            _task_meta(
                dataset_name="small_clf",
                task_id_str="1",
                num_features=10,
                num_classes=3,
                splits=[_split_meta(fold=0, num_instances_train=500)],
            ),
            _task_meta(
                dataset_name="huge_clf",
                task_id_str="2",
                num_features=600,  # too many features for tabpfn
                num_classes=3,
                splits=[_split_meta(fold=0, num_instances_train=50_000)],
            ),
        ]
        df = to_legacy_task_metadata(metas)
        # `n_samples_train_per_fold` is what the size predicates alias to `max_train_rows`.
        small = df[df["n_samples_train_per_fold"] <= 10_000]
        assert set(small["dataset"]) == {"small_clf"}
        tabpfn = df[(df["n_samples_train_per_fold"] <= 10_000) & (df["n_features"] <= 500) & (df["n_classes"] <= 10)]
        assert set(tabpfn["dataset"]) == {"small_clf"}
