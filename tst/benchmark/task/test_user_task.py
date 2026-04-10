from __future__ import annotations

import functools
import operator
from pathlib import Path

import numpy as np
import openml
import pandas as pd
import pytest
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask, TaskType
from openml.tasks.split import OpenMLSplit
from tabarena.benchmark.task import UserTask
from tabarena.benchmark.task.user_task import (
    GroupLabelTypes,
    SplitMetadata,
    TabArenaTaskMetadata,
    TabArenaTaskMetadataMixin,
    from_sklearn_splits_to_user_task_splits,
)


@pytest.fixture(scope="session", autouse=True)
def _isolate_openml_cache(tmp_path_factory):
    tmp_cache = tmp_path_factory.mktemp("openml_cache")
    openml.config.set_root_cache_directory(tmp_cache)
    Path(openml.config._root_cache_directory).mkdir(parents=True, exist_ok=True)


def _make_dataset(problem_type: str, *, n: int = 10) -> tuple[pd.DataFrame, str, list[str] | None, list[bool]]:
    dataset = pd.DataFrame(
        {
            "num": np.arange(n, dtype="int64"),
            "cat": pd.Series(["A", "B"] * (n // 2), dtype="category"),
        }
    )
    if problem_type == "classification":
        dataset["target"] = ["neg", "pos"] * (n // 2)
        dataset["target"] = dataset["target"].astype("category")
        class_labels = ["neg", "pos"]
    else:  # regression
        dataset["target"] = np.linspace(0.0, 1.0, num=n)
        class_labels = None

    cat_indicator = [False, True]
    return dataset, "target", class_labels, cat_indicator


@pytest.mark.parametrize(
    ("problem_type", "expected_cls"),
    [
        ("classification", OpenMLClassificationTask),
        ("regression", OpenMLRegressionTask),
    ],
)
def test_user_task_as_openml_task(problem_type, expected_cls, tmp_path):
    """Test that UserTask can be converted to an OpenML task for local use.
    This does not test the splits, which are tested in another test.
    """
    df_original, target_feature, _class_labels, cat_indicator = _make_dataset(problem_type, n=10)
    splits = {0: {0: (list(range(8)), [8, 9])}}

    ut = UserTask(
        task_name=f"unit-test-{problem_type}",
        task_cache_path=tmp_path,
    )
    oml_task = ut.create_local_openml_task(
        dataset=df_original,
        target_feature=target_feature,
        problem_type=problem_type,
        splits=splits,
    )

    # Check Task Metadata
    assert isinstance(oml_task, expected_cls), f"Expected {expected_cls}, got {type(oml_task)}"
    if problem_type == "classification":
        assert oml_task.task_type_id == TaskType.SUPERVISED_CLASSIFICATION
        assert oml_task.class_labels == ["neg", "pos"]
    else:
        assert oml_task.task_type_id == TaskType.SUPERVISED_REGRESSION
    assert oml_task.task_id == ut.task_id
    assert oml_task.dataset_id == -1
    assert oml_task.task_type == "None"
    assert oml_task.target_name == target_feature

    # Check Dataset Metadata
    oml_dataset = oml_task.get_dataset()
    assert isinstance(oml_dataset, openml.datasets.OpenMLDataset)
    assert oml_dataset.name == ut.get_dataset_name()
    assert oml_dataset.default_target_attribute == target_feature
    assert oml_dataset.parquet_file == (ut._local_cache_path / "data.pq")
    assert (ut._local_cache_path / "data.pq").exists()
    assert oml_dataset.data_file == "ignored"

    # Check Dataset State
    X, y, categorical_indicator, attribute_names = oml_dataset.get_data(target=oml_task.target_name)
    assert categorical_indicator == cat_indicator
    expected_a_names = list(df_original.columns)
    expected_a_names.remove(target_feature)
    assert attribute_names == expected_a_names
    X[target_feature] = y
    pd.testing.assert_frame_equal(
        X.sort_index(axis=1),
        df_original.sort_index(axis=1),
        check_dtype=False,
    )

    # Check Split State
    assert isinstance(oml_task.split, OpenMLSplit)
    expected_split = OpenMLSplit(
        name="User-Splits",
        description="User-defined splits for a custom task.",
        split={r: {f: {0: (np.array(tr), np.array(te))} for f, (tr, te) in splits[r].items()} for r in splits},
    )
    assert oml_task.split == expected_split


@pytest.mark.parametrize(
    ("splits", "n_samples"),
    [
        # 1-repeat / 1-fold – the absolute minimum
        (
            {
                0: {0: ([0, 1], [2, 3])},
            },
            4,
        ),
        # 2-repeat / 2-fold, identical structure, no overlaps
        (
            {
                0: {
                    0: ([0, 1], [4, 5]),
                    1: ([2, 3], [6, 7]),
                },
                1: {
                    0: ([4, 5], [0, 1]),
                    1: ([6, 7], [2, 3]),
                },
            },
            8,
        ),
    ],
    ids=["minimal", "multi_repeat_multi_fold"],
)
def test_validate_splits_valid(splits, n_samples):
    """No exception is expected for well-formed splits."""
    UserTask._validate_splits(splits=splits, n_samples=n_samples)


@pytest.mark.parametrize(
    ("splits", "n_samples", "exc_regex"),
    [
        # Not a dict at all
        ("not a dict", 4, r"Splits must be a dictionary"),
        # Repeat entry not a dict
        ({0: "oops"}, 4, r"repeat 0 must be a dictionary"),
        # Train / test containers not lists
        ({0: {0: ((0, 1), [2])}}, 4, r"split 0 must be lists"),
        # Non-integer index
        ({0: {0: ([0.0], [1])}}, 2, r"indices .* must be integers"),
        # Empty train list
        ({0: {0: ([], [1])}}, 2, r"must not be empty"),
        # Overlap between train & test
        ({0: {0: ([0, 1], [1, 2])}}, 3, r"must not overlap"),
        # Negative index
        ({0: {0: ([-1], [1])}}, 3, r"must be non-negative"),
        # Index >= n_samples
        ({0: {0: ([0], [3])}}, 3, r"must not exceed the dataset size"),
        # Overlap of test indices across folds in same repeat
        (
            {0: {0: ([0], [1]), 1: ([2], [1])}},
            3,
            r"must not overlap with previous splits in repeat 0",
        ),
        # Different number of folds across repeats
        (
            {0: {0: ([0], [1])}, 1: {0: ([1], [0]), 1: ([0], [1])}},
            3,
            r"All repeats must have the same number of splits",
        ),
    ],
    ids=[
        "splits_not_dict",
        "repeat_not_dict",
        "indices_not_lists",
        "non_integer_index",
        "empty_train",
        "train_test_overlap",
        "negative_index",
        "index_out_of_bounds",
        "test_overlap_across_folds",
        "unequal_folds_across_repeats",
    ],
)
def test_validate_splits_invalid(splits, n_samples, exc_regex):
    """Every malformed split configuration should raise and emit the right message."""
    with pytest.raises(ValueError, match=exc_regex):
        UserTask._validate_splits(splits=splits, n_samples=n_samples)


# ---------------------------------------------------------------------------
# Helpers for new tests
# ---------------------------------------------------------------------------


def _make_split_metadata(
    repeat: int = 0,
    fold: int = 0,
    n_train: int = 8,
    n_test: int = 2,
) -> SplitMetadata:
    return SplitMetadata(
        repeat=repeat,
        fold=fold,
        num_instances_train=n_train,
        num_instances_test=n_test,
        num_instance_groups_train=n_train,
        num_instance_groups_test=n_test,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=2,
        num_features_test=2,
    )


def _make_task_metadata(
    splits_metadata: dict | None = None,
) -> TabArenaTaskMetadata:
    if splits_metadata is None:
        s = _make_split_metadata()
        splits_metadata = {s.split_index: s}
    return TabArenaTaskMetadata(
        dataset_name="test-dataset",
        problem_type="binary",
        is_classification=True,
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
        multiclass_min_n_classes_over_splits=2,
        multiclass_max_n_classes_over_splits=2,
        class_consistency_over_splits=True,
        num_instances=10,
        num_features=2,
        num_classes=2,
        num_instance_groups=10,
        tabarena_task_name="test-task",
        task_id_str="UserTask|1234567890|test-task|/tmp/cache",
    )


# ---------------------------------------------------------------------------
# UserTask property tests
# ---------------------------------------------------------------------------


def test_task_id_is_deterministic():
    assert UserTask(task_name="hello").task_id == UserTask(task_name="hello").task_id


def test_task_id_differs_for_different_names():
    assert UserTask(task_name="a").task_id != UserTask(task_name="b").task_id


def test_task_id_is_non_negative_int_below_10e10():
    task_id = UserTask(task_name="test").task_id
    assert isinstance(task_id, int)
    assert 0 <= task_id < 10**10


def test_task_id_str_format(tmp_path):
    ut = UserTask(task_name="my-task", task_cache_path=tmp_path)
    parts = ut.task_id_str.split("|")
    assert parts[0] == "UserTask"
    assert int(parts[1]) == ut.task_id
    assert parts[2] == "my-task"
    assert Path(parts[3]) == tmp_path


def test_tabarena_task_name():
    ut = UserTask(task_name="my-task")
    assert ut.tabarena_task_name == f"Task-{ut.task_id}"


def test_from_task_id_str_round_trip(tmp_path):
    ut = UserTask(task_name="round-trip", task_cache_path=tmp_path)
    ut2 = UserTask.from_task_id_str(ut.task_id_str)
    assert ut2.task_name == ut.task_name
    assert ut2.task_cache_path == ut.task_cache_path


@pytest.mark.parametrize(
    "bad_str",
    [
        "bad",
        "UserTask|123|name",
        "NotUserTask|123|name|/tmp",
    ],
    ids=["too_short", "missing_path", "wrong_prefix"],
)
def test_from_task_id_str_invalid(bad_str):
    with pytest.raises(ValueError, match="Invalid task ID string"):
        UserTask.from_task_id_str(bad_str)


def test_get_dataset_name_default():
    ut = UserTask(task_name="my-task")
    assert ut.get_dataset_name() == "Dataset-my-task"


def test_get_dataset_name_with_explicit_name():
    ut = UserTask(task_name="my-task")
    assert ut.get_dataset_name("CustomName") == "CustomName"


def test_task_cache_path_custom(tmp_path):
    ut = UserTask(task_name="test", task_cache_path=tmp_path)
    assert ut.task_cache_path == tmp_path


def test_task_cache_path_default_contains_tabarena_tasks():
    ut = UserTask(task_name="test")
    assert "tabarena_tasks" in str(ut.task_cache_path)


# ---------------------------------------------------------------------------
# UserTask save / load round-trip
# ---------------------------------------------------------------------------


def test_save_load_round_trip_classification(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=10)
    splits = {0: {0: (list(range(8)), [8, 9])}}
    ut = UserTask(task_name="save-load-clf", task_cache_path=tmp_path)
    task = ut.create_local_openml_task(dataset=df, target_feature=target, problem_type="classification", splits=splits)
    ut.save_local_openml_task(task)
    assert ut.openml_task_path.exists()

    loaded = ut.load_local_openml_task()
    assert loaded.task_id == ut.task_id
    assert loaded.target_name == target
    # get_dataset patch must be re-applied
    X, _y, _, _ = loaded.get_dataset().get_data(target=loaded.target_name)
    assert len(X) == 10


def test_save_load_round_trip_regression(tmp_path):
    df, target, _, _ = _make_dataset("regression", n=10)
    splits = {0: {0: (list(range(8)), [8, 9])}}
    ut = UserTask(task_name="save-load-reg", task_cache_path=tmp_path)
    task = ut.create_local_openml_task(dataset=df, target_feature=target, problem_type="regression", splits=splits)
    ut.save_local_openml_task(task)
    loaded = ut.load_local_openml_task()
    assert loaded.task_id == ut.task_id


def test_load_nonexistent_task_raises(tmp_path):
    ut = UserTask(task_name="does-not-exist", task_cache_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        ut.load_local_openml_task()


# ---------------------------------------------------------------------------
# Multi-fold task creation
# ---------------------------------------------------------------------------


def test_create_local_openml_task_multi_repeat_multi_fold(tmp_path):
    n = 20
    df, target, _, _ = _make_dataset("classification", n=n)
    splits = {
        0: {
            0: (list(range(15)), list(range(15, 20))),
            1: (list(range(5, 20)), list(range(5))),
        },
        1: {
            0: (list(range(10, 20)) + list(range(5)), list(range(5, 10))),
            1: (list(range(10)) + list(range(15, 20)), list(range(10, 15))),
        },
    }
    ut = UserTask(task_name="multi-fold", task_cache_path=tmp_path)
    task = ut.create_local_openml_task(dataset=df, target_feature=target, problem_type="classification", splits=splits)
    # Two repeats, two folds each
    assert set(task.split.split.keys()) == {0, 1}
    assert set(task.split.split[0].keys()) == {0, 1}
    assert set(task.split.split[1].keys()) == {0, 1}
    # Spot-check one split's train indices
    train_arr, test_arr = task.split.split[0][0][0]
    assert train_arr.tolist() == list(range(15))
    assert test_arr.tolist() == list(range(15, 20))


# ---------------------------------------------------------------------------
# SplitMetadata tests
# ---------------------------------------------------------------------------


def test_split_metadata_get_split_index():
    assert SplitMetadata.get_split_index(repeat_i=0, fold_i=0) == "r0f0"
    assert SplitMetadata.get_split_index(repeat_i=2, fold_i=5) == "r2f5"


def test_split_metadata_split_index_property():
    s = _make_split_metadata(repeat=3, fold=7)
    assert s.split_index == "r3f7"


def test_split_metadata_to_dict_contains_split_index():
    s = _make_split_metadata(repeat=1, fold=2)
    d = s.to_dict()
    assert d["split_index"] == "r1f2"
    assert d["repeat"] == 1
    assert d["fold"] == 2
    assert d["num_instances_train"] == 8


# ---------------------------------------------------------------------------
# TabArenaTaskMetadata tests
# ---------------------------------------------------------------------------


def test_task_metadata_n_splits_single():
    assert _make_task_metadata().n_splits == 1


def test_task_metadata_n_splits_multi():
    s0 = _make_split_metadata(repeat=0, fold=0)
    s1 = _make_split_metadata(repeat=0, fold=1)
    meta = _make_task_metadata({s0.split_index: s0, s1.split_index: s1})
    assert meta.n_splits == 2


def test_task_metadata_split_indices():
    meta = _make_task_metadata()
    assert meta.split_indices == ["r0f0"]


def test_task_metadata_split_index_single():
    assert _make_task_metadata().split_index == "r0f0"


def test_task_metadata_split_index_multi_raises():
    s0 = _make_split_metadata(repeat=0, fold=0)
    s1 = _make_split_metadata(repeat=0, fold=1)
    meta = _make_task_metadata({s0.split_index: s0, s1.split_index: s1})
    with pytest.raises(ValueError, match="2 splits"):
        _ = meta.split_index


def test_task_metadata_to_dict_includes_splits_metadata():
    d = _make_task_metadata().to_dict()
    assert "splits_metadata" in d
    assert d["dataset_name"] == "test-dataset"


def test_task_metadata_to_dict_exclude_splits_metadata():
    d = _make_task_metadata().to_dict(exclude_splits_metadata=True)
    assert "splits_metadata" not in d
    assert d["problem_type"] == "binary"


def test_task_metadata_to_dataframe_shape():
    s0 = _make_split_metadata(repeat=0, fold=0)
    s1 = _make_split_metadata(repeat=0, fold=1)
    meta = _make_task_metadata({s0.split_index: s0, s1.split_index: s1})
    df = meta.to_dataframe()
    assert len(df) == 2
    assert "split_index" in df.columns
    assert "dataset_name" in df.columns
    assert set(df["split_index"]) == {"r0f0", "r0f1"}


def test_task_metadata_from_row_round_trip():
    meta = _make_task_metadata()
    df = meta.to_dataframe()
    reconstructed = TabArenaTaskMetadata.from_row(df.iloc[0])
    assert reconstructed.dataset_name == meta.dataset_name
    assert reconstructed.problem_type == meta.problem_type
    assert reconstructed.n_splits == 1
    assert reconstructed.split_index == meta.split_index
    assert reconstructed.splits_metadata["r0f0"].num_instances_train == 8


def test_task_metadata_from_row_missing_field_raises():
    meta = _make_task_metadata()
    df = meta.to_dataframe()
    row = df.iloc[0].drop("dataset_name")
    with pytest.raises(ValueError, match="missing required TabArenaTaskMetadata fields"):
        TabArenaTaskMetadata.from_row(row)


def test_task_metadata_unroll_splits():
    s0 = _make_split_metadata(repeat=0, fold=0)
    s1 = _make_split_metadata(repeat=1, fold=0)
    meta = _make_task_metadata({s0.split_index: s0, s1.split_index: s1})
    unrolled = meta.unroll_splits()
    assert len(unrolled) == 2
    for u in unrolled:
        assert u.n_splits == 1
    assert unrolled[0].split_index == "r0f0"
    assert unrolled[1].split_index == "r1f0"
    # Static fields are preserved
    assert unrolled[0].dataset_name == meta.dataset_name


def test_task_metadata_from_row_backward_compat_missing_optional_fields():
    """Old CSVs without the new dtype flag columns should load without error."""
    meta = _make_task_metadata()
    df = meta.to_dataframe()
    row = df.iloc[0]
    # Simulate an old CSV that doesn't have the new columns
    row = row.drop(["has_datetime", "has_text", "has_categorical", "has_numeric"])
    reconstructed = TabArenaTaskMetadata.from_row(row)
    assert reconstructed.dataset_name == meta.dataset_name
    # New fields default to None when absent
    assert reconstructed.has_datetime is None
    assert reconstructed.has_text is None
    assert reconstructed.has_categorical is None
    assert reconstructed.has_numeric is None


def test_task_metadata_dtype_flags_default_to_none():
    """New dtype flag fields default to None when not passed explicitly."""
    meta = _make_task_metadata()
    assert meta.has_datetime is None
    assert meta.has_text is None
    assert meta.has_categorical is None
    assert meta.has_numeric is None


def test_task_metadata_dtype_flags_round_trip():
    """Dtype flags survive a to_dataframe / from_row round trip."""
    s = _make_split_metadata()
    meta = TabArenaTaskMetadata(
        dataset_name="test",
        problem_type="binary",
        is_classification=True,
        target_name="target",
        eval_metric="roc_auc",
        splits_metadata={s.split_index: s},
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
        num_instances=10,
        num_features=2,
        num_classes=2,
        num_instance_groups=10,
        tabarena_task_name="test",
        task_id_str=None,
        has_datetime=False,
        has_text=True,
        has_categorical=True,
        has_numeric=False,
    )
    df = meta.to_dataframe()
    reconstructed = TabArenaTaskMetadata.from_row(df.iloc[0])
    assert reconstructed.has_datetime is False
    assert reconstructed.has_text is True
    assert reconstructed.has_categorical is True
    assert reconstructed.has_numeric is False


# ---------------------------------------------------------------------------
# from_sklearn_splits_to_user_task_splits
# ---------------------------------------------------------------------------


def test_from_sklearn_splits_single_repeat():
    sklearn_splits = [
        (np.array([2, 3, 4]), np.array([0, 1])),
        (np.array([0, 1, 4]), np.array([2, 3])),
    ]
    result = from_sklearn_splits_to_user_task_splits(sklearn_splits, n_splits=2)
    assert set(result.keys()) == {0}
    assert set(result[0].keys()) == {0, 1}
    assert result[0][0] == ([2, 3, 4], [0, 1])
    assert result[0][1] == ([0, 1, 4], [2, 3])


def test_from_sklearn_splits_multiple_repeats():
    sklearn_splits = [
        (np.array([2, 3]), np.array([0, 1])),
        (np.array([0, 1]), np.array([2, 3])),
        (np.array([1, 3]), np.array([0, 2])),
        (np.array([0, 2]), np.array([1, 3])),
    ]
    result = from_sklearn_splits_to_user_task_splits(sklearn_splits, n_splits=2)
    assert set(result.keys()) == {0, 1}
    assert set(result[0].keys()) == {0, 1}
    assert set(result[1].keys()) == {0, 1}
    assert result[1][0] == ([1, 3], [0, 2])


# ---------------------------------------------------------------------------
# TabArenaTaskMetadataMixin.get_num_instance_groups
# ---------------------------------------------------------------------------


def test_get_num_instance_groups_no_group():
    X = pd.DataFrame({"a": [1, 2, 3]})
    n = TabArenaTaskMetadataMixin.get_num_instance_groups(X=X, group_on=None, group_labels=None)
    assert n == 3


def test_get_num_instance_groups_per_sample_label():
    X = pd.DataFrame({"a": [1, 2, 3], "group": ["x", "x", "y"]})
    n = TabArenaTaskMetadataMixin.get_num_instance_groups(
        X=X, group_on="group", group_labels=GroupLabelTypes.PER_SAMPLE
    )
    # PER_SAMPLE → returns len(X) regardless of groups
    assert n == 3


def test_get_num_instance_groups_per_group_label():
    X = pd.DataFrame({"a": [1, 2, 3, 4], "group": ["x", "x", "y", "y"]})
    n = TabArenaTaskMetadataMixin.get_num_instance_groups(X=X, group_on="group", group_labels=GroupLabelTypes.PER_GROUP)
    assert n == 2


def test_get_num_instance_groups_multi_column_group():
    X = pd.DataFrame({"g1": ["a", "a", "b"], "g2": [1, 2, 1]})
    n = TabArenaTaskMetadataMixin.get_num_instance_groups(
        X=X, group_on=["g1", "g2"], group_labels=GroupLabelTypes.PER_GROUP
    )
    # (a,1), (a,2), (b,1) → 3 unique groups
    assert n == 3


# ---------------------------------------------------------------------------
# Helpers for _get_dataset_stats / compute_metadata tests
# ---------------------------------------------------------------------------


def _make_multiclass_dataset(n_per_class: int = 4) -> tuple[pd.DataFrame, str]:
    """Create a 3-class classification dataset with n_per_class samples per class."""
    n = n_per_class * 3
    labels = (["cls0"] * n_per_class) + (["cls1"] * n_per_class) + (["cls2"] * n_per_class)
    df = pd.DataFrame(
        {
            "num": np.arange(n, dtype="int64"),
            "cat": pd.Categorical(["A", "B"] * (n // 2)),
            "target": pd.Categorical(labels),
        }
    )
    return df, "target"


def _make_4class_dataset(n_per_class: int = 3) -> tuple[pd.DataFrame, str]:
    """Create a 4-class dataset where fold splits can yield different class counts."""
    n = n_per_class * 4
    labels = functools.reduce(operator.iadd, ([f"cls{c}"] * n_per_class for c in range(4)), [])
    df = pd.DataFrame(
        {
            "num": np.arange(n, dtype="int64"),
            "cat": pd.Categorical(["A", "B"] * (n // 2)),
            "target": pd.Categorical(labels),
        }
    )
    return df, "target"


def _task_from_user_task(
    df: pd.DataFrame,
    target: str,
    problem_type: str,
    splits: dict,
    tmp_path: Path,
    task_name: str = "test-task",
    **kwargs,
):
    ut = UserTask(task_name=task_name, task_cache_path=tmp_path)
    task = ut.create_local_openml_task(
        dataset=df,
        target_feature=target,
        problem_type=problem_type,
        splits=splits,
        **kwargs,
    )
    return task, ut


# ---------------------------------------------------------------------------
# _get_dataset_stats
# ---------------------------------------------------------------------------


def test_get_dataset_stats_regression_basic(tmp_path):
    """Regression: num_classes=-1, num_features excludes target, num_instance_groups==len."""
    df, target, _, _ = _make_dataset("regression", n=10)
    task, _ = _task_from_user_task(df, target, "regression", {0: {0: (list(range(8)), [8, 9])}}, tmp_path, "ds-reg")
    n_inst, n_feat, n_cls, n_groups = task._get_dataset_stats(
        oml_dataset=df, is_classification=False, target_name=target
    )
    assert n_inst == 10
    assert n_feat == 2  # "num" and "cat"
    assert n_cls == -1
    assert n_groups == 10  # group_on is None → equals len(df)


def test_get_dataset_stats_classification_class_count(tmp_path):
    """Classification: num_classes equals the unique target count in the slice."""
    df, target, _, _ = _make_dataset("classification", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "ds-clf",
    )
    _, _, n_cls, _ = task._get_dataset_stats(oml_dataset=df, is_classification=True, target_name=target)
    assert n_cls == 2


def test_get_dataset_stats_num_features_excludes_target(tmp_path):
    """5-column frame (4 features + target) → num_features == 4."""
    n = 10
    df = pd.DataFrame(
        {
            "f1": range(n),
            "f2": range(n),
            "f3": range(n),
            "f4": range(n),
            "target": [0, 1] * (n // 2),
        }
    )
    task, _ = _task_from_user_task(
        df,
        "target",
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "ds-5col",
    )
    _, n_feat, _, _ = task._get_dataset_stats(oml_dataset=df, is_classification=True, target_name="target")
    assert n_feat == 4


def test_get_dataset_stats_slice_reports_subset_class_count(tmp_path):
    """Passing a subset of rows reports the class count *in that slice*."""
    df, target, _, _ = _make_dataset("classification", n=10)
    # First 5 rows: alternating neg/pos → 2 classes
    # We make a subset that has only one class to check that it really counts the slice.
    subset_one_class = df[df["target"] == "neg"].copy()
    task, _ = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "ds-slice",
    )
    _, _, n_cls, _ = task._get_dataset_stats(oml_dataset=subset_one_class, is_classification=True, target_name=target)
    assert n_cls == 1


# ---------------------------------------------------------------------------
# compute_metadata — regression
# ---------------------------------------------------------------------------


def test_compute_metadata_regression(tmp_path):
    df, target, _, _ = _make_dataset("regression", n=10)
    task, ut = _task_from_user_task(df, target, "regression", {0: {0: (list(range(8)), [8, 9])}}, tmp_path, "cm-reg")
    meta = task.compute_metadata(tabarena_task_name="my-task", task_id_str=ut.task_id_str)

    assert meta.problem_type == "regression"
    assert meta.is_classification is False
    assert meta.num_classes == -1
    assert meta.multiclass_min_n_classes_over_splits is None
    assert meta.multiclass_max_n_classes_over_splits is None
    assert meta.class_consistency_over_splits is None
    assert meta.tabarena_task_name == "my-task"
    assert meta.task_id_str == ut.task_id_str


def test_compute_metadata_regression_split_stats(tmp_path):
    df, target, _, _ = _make_dataset("regression", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "regression",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-reg-split",
    )
    meta = task.compute_metadata()

    assert meta.n_splits == 1
    split = meta.splits_metadata["r0f0"]
    assert split.num_instances_train == 8
    assert split.num_instances_test == 2
    assert split.num_classes_train == -1
    assert split.num_classes_test == -1
    assert split.num_features_train == 2
    assert split.num_features_test == 2


# ---------------------------------------------------------------------------
# compute_metadata — binary classification
# ---------------------------------------------------------------------------


def test_compute_metadata_binary_problem_type(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-bin",
    )
    meta = task.compute_metadata()

    assert meta.problem_type == "binary"
    assert meta.is_classification is True
    assert meta.num_classes == 2
    assert meta.multiclass_min_n_classes_over_splits == 2
    assert meta.multiclass_max_n_classes_over_splits == 2
    assert meta.class_consistency_over_splits is True


def test_compute_metadata_binary_dataset_level_stats(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=10)
    task, ut = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-bin-ds",
    )
    meta = task.compute_metadata()

    assert meta.dataset_name == ut.get_dataset_name()
    assert meta.target_name == target
    assert meta.num_instances == 10
    assert meta.num_features == 2  # "num" and "cat"
    assert meta.num_instance_groups == 10  # no group_on


def test_compute_metadata_dtype_flags(tmp_path):
    """_make_dataset produces int64 'num' and category 'cat' feature columns."""
    df, target, _, _ = _make_dataset("classification", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-dtype-flags",
    )
    meta = task.compute_metadata()

    assert meta.has_numeric is True
    assert meta.has_categorical is True
    assert meta.has_datetime is False
    assert meta.has_text is False


def test_compute_metadata_dtype_flags_with_text_and_datetime(tmp_path):
    df = pd.DataFrame(
        {
            "num": np.arange(10, dtype="float64"),
            "txt": pd.array(["hello"] * 10, dtype="string"),
            "dt": pd.date_range("2020-01-01", periods=10, freq="D"),
            "target": [0, 1] * 5,
        }
    )
    task, _ = _task_from_user_task(
        df,
        "target",
        "classification",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-dtype-all",
    )
    meta = task.compute_metadata()

    assert meta.has_numeric is True
    assert meta.has_text is True
    assert meta.has_datetime is True
    assert meta.has_categorical is False


# ---------------------------------------------------------------------------
# compute_metadata — multiclass classification
# ---------------------------------------------------------------------------


def test_compute_metadata_multiclass_problem_type(tmp_path):
    df, target = _make_multiclass_dataset(n_per_class=4)  # 12 samples, 3 classes
    task, _ = _task_from_user_task(
        df,
        target,
        "classification",
        {0: {0: (list(range(9)), list(range(9, 12)))}},
        tmp_path,
        "cm-multi",
    )
    meta = task.compute_metadata()

    assert meta.problem_type == "multiclass"
    assert meta.num_classes == 3
    assert meta.multiclass_min_n_classes_over_splits == 3
    assert meta.multiclass_max_n_classes_over_splits == 3
    assert meta.class_consistency_over_splits is True


def test_compute_metadata_class_consistency_false(tmp_path):
    """Two repeats where each sees a different number of classes → class_consistency=False.

    Dataset has 4 classes (cls0–cls3), 8 samples each, n=32.
    Use 2 repeats (1 fold each) so the repeats are fully independent:

      Repeat 0, Fold 0:
        train = [0..23]          → cls0, cls1, cls2 only (3 classes)
        test  = [24..31]         → cls3 only (1 class)
        max(3, 1) = 3 → multiclass, appends 3

      Repeat 1, Fold 0:
        train = first-half of every class  → 4 classes
        test  = second-half of every class → 4 classes
        max(4, 4) = 4 → multiclass, appends 4

    num_classes_list = [3, 4] → class_consistency_over_splits = False.
    """
    df, target = _make_4class_dataset(n_per_class=8)  # 32 samples
    # Repeat 1 train/test: alternate halves of each 8-sample class block.
    r1_train = list(range(4)) + list(range(8, 12)) + list(range(16, 20)) + list(range(24, 28))
    r1_test = list(range(4, 8)) + list(range(12, 16)) + list(range(20, 24)) + list(range(28, 32))
    splits = {
        0: {0: (list(range(24)), list(range(24, 32)))},
        1: {0: (r1_train, r1_test)},
    }
    task, _ = _task_from_user_task(df, target, "classification", splits, tmp_path, "cm-inconsistent")
    meta = task.compute_metadata()

    assert meta.problem_type == "multiclass"
    assert meta.class_consistency_over_splits is False
    assert meta.multiclass_min_n_classes_over_splits == 3
    assert meta.multiclass_max_n_classes_over_splits == 4


# ---------------------------------------------------------------------------
# compute_metadata — multi-fold and multi-repeat
# ---------------------------------------------------------------------------


def test_compute_metadata_multi_fold_split_indices(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=20)
    splits = {
        0: {
            0: (list(range(15)), list(range(15, 20))),
            1: (list(range(5, 20)), list(range(5))),
        }
    }
    task, _ = _task_from_user_task(df, target, "classification", splits, tmp_path, "cm-mf")
    meta = task.compute_metadata()

    assert meta.n_splits == 2
    assert set(meta.split_indices) == {"r0f0", "r0f1"}


def test_compute_metadata_multi_fold_per_split_counts(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=20)
    splits = {
        0: {
            0: (list(range(15)), list(range(15, 20))),
            1: (list(range(5, 20)), list(range(5))),
        }
    }
    task, _ = _task_from_user_task(df, target, "classification", splits, tmp_path, "cm-mf-cnt")
    meta = task.compute_metadata()

    s0 = meta.splits_metadata["r0f0"]
    assert s0.num_instances_train == 15
    assert s0.num_instances_test == 5
    s1 = meta.splits_metadata["r0f1"]
    assert s1.num_instances_train == 15
    assert s1.num_instances_test == 5


def test_compute_metadata_multi_repeat_split_indices(tmp_path):
    df, target, _, _ = _make_dataset("classification", n=20)
    splits = {
        0: {0: (list(range(15)), list(range(15, 20)))},
        1: {0: (list(range(5, 20)), list(range(5)))},
    }
    task, _ = _task_from_user_task(df, target, "classification", splits, tmp_path, "cm-mr")
    meta = task.compute_metadata()

    assert meta.n_splits == 2
    assert "r0f0" in meta.splits_metadata
    assert "r1f0" in meta.splits_metadata


# ---------------------------------------------------------------------------
# compute_metadata — optional fields passthrough
# ---------------------------------------------------------------------------


def test_compute_metadata_optional_fields_default_none(tmp_path):
    df, target, _, _ = _make_dataset("regression", n=10)
    task, _ = _task_from_user_task(df, target, "regression", {0: {0: (list(range(8)), [8, 9])}}, tmp_path, "cm-opt")
    meta = task.compute_metadata()

    assert meta.tabarena_task_name is None
    assert meta.task_id_str is None
    assert meta.stratify_on is None
    assert meta.group_on is None
    assert meta.time_on is None
    assert meta.group_time_on is None
    assert meta.group_labels is None
    assert meta.split_time_horizon is None
    assert meta.split_time_horizon_unit is None
    assert meta.eval_metric is None


def test_compute_metadata_stores_result_on_task(tmp_path):
    """compute_metadata stores result in task._task_metadata."""
    df, target, _, _ = _make_dataset("regression", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "regression",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-store",
    )
    assert task._task_metadata is None
    meta = task.compute_metadata()
    assert task._task_metadata is meta


def test_compute_metadata_split_time_horizon_passthrough(tmp_path):
    df, target, _, _ = _make_dataset("regression", n=10)
    task, _ = _task_from_user_task(
        df,
        target,
        "regression",
        {0: {0: (list(range(8)), [8, 9])}},
        tmp_path,
        "cm-horizon",
    )
    # Patch the attributes directly — they are set via __init__ on the mixin.
    task.split_time_horizon = 30
    task.split_time_horizon_unit = "days"
    meta = task.compute_metadata()

    assert meta.split_time_horizon == 30
    assert meta.split_time_horizon_unit == "days"


# ---------------------------------------------------------------------------
# Unsupported ARFF dtype workaround (datetime64, timedelta64, period, interval)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("col_name", "col_values", "dtype"),
    [
        (
            "date_col",
            pd.to_datetime(["2020-01-01", "2021-06-15"] * 5),
            "datetime64[ns]",
        ),
        (
            "delta_col",
            pd.to_timedelta(range(10), unit="D"),
            "timedelta64[ns]",
        ),
        (
            "period_col",
            pd.period_range("2020-01", periods=10, freq="M"),
            pd.PeriodDtype(freq="M"),
        ),
        (
            "interval_col",
            pd.arrays.IntervalArray.from_breaks(range(11)),
            pd.IntervalDtype(subtype="int64", closed="right"),
        ),
        # Note: complex128 is also unsupported by liac-arff, but pyarrow (parquet)
        # cannot serialize it either, so it fails at a later stage and is excluded here.
    ],
    ids=["datetime64", "timedelta64", "period", "interval"],
)
def test_create_local_openml_task_unsupported_arff_dtype_does_not_raise(col_name, col_values, dtype, tmp_path):
    """Columns with dtypes unsupported by liac-arff (datetime64, timedelta64, complex)
    must not prevent task creation — they are cast to string only for ARFF attribute
    inference and do not affect the data persisted to parquet.
    """
    n = 10
    df = pd.DataFrame(
        {
            "num": np.arange(n, dtype="int64"),
            col_name: col_values,
            "target": np.linspace(0.0, 1.0, num=n),
        }
    )
    assert df[col_name].dtype == dtype

    splits = {0: {0: (list(range(8)), [8, 9])}}
    ut = UserTask(task_name=f"unsupported-dtype-{col_name}", task_cache_path=tmp_path)
    # Must not raise
    ut.create_local_openml_task(dataset=df, target_feature="target", problem_type="regression", splits=splits)

    # The parquet file must store the original dtype — the workaround must not
    # modify the persisted data.
    stored = pd.read_parquet(ut._local_cache_path / "data.pq")
    assert stored[col_name].dtype == df[col_name].dtype


@pytest.mark.parametrize(
    ("cat_values", "cat_dtype", "test_id"),
    [
        (
            pd.Categorical([0, 1, 2, 1, 0, 2, 1, 0, 2, 1]),
            "int64",
            "int_categories",
        ),
        (
            pd.Categorical([1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5, 1.5, 2.5]),
            "float64",
            "float_categories",
        ),
    ],
    ids=lambda x: x if isinstance(x, str) and "_categories" in x else "",
)
def test_create_local_openml_task_non_string_categorical_does_not_raise(cat_values, cat_dtype, test_id, tmp_path):
    """Categorical columns whose categories have a non-string dtype (e.g. int, float)
    must not break ARFF attribute inference — categories are cast to string only for
    metadata and do not affect the data persisted to parquet.
    """
    n = 10
    df = pd.DataFrame(
        {
            "num": np.arange(n, dtype="int64"),
            "cat_col": cat_values,
            "target": np.linspace(0.0, 1.0, num=n),
        }
    )
    assert df["cat_col"].dtype.name == "category"
    assert df["cat_col"].cat.categories.dtype == cat_dtype

    splits = {0: {0: (list(range(8)), [8, 9])}}
    ut = UserTask(task_name=f"non-str-cat-{test_id}", task_cache_path=tmp_path)
    # Must not raise
    ut.create_local_openml_task(dataset=df, target_feature="target", problem_type="regression", splits=splits)

    # The parquet file must store the original data values unchanged.
    stored = pd.read_parquet(ut._local_cache_path / "data.pq")
    pd.testing.assert_series_equal(stored["cat_col"].astype(str), df["cat_col"].astype(str), check_names=True)
