from __future__ import annotations

from pathlib import Path

import numpy as np
import openml
import pandas as pd
import pytest
from openml.tasks import OpenMLClassificationTask, OpenMLRegressionTask, TaskType
from openml.tasks.split import OpenMLSplit
from tabarena.benchmark.task import UserTask


@pytest.fixture(scope="session", autouse=True)
def _isolate_openml_cache(tmp_path_factory):
    tmp_cache = tmp_path_factory.mktemp("openml_cache")
    openml.config.set_root_cache_directory(tmp_cache)
    Path(openml.config._root_cache_directory).mkdir(parents=True, exist_ok=True)


def _make_dataset(
    problem_type: str, *, n: int = 10
) -> tuple[pd.DataFrame, str, list[str] | None, list[bool]]:
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
    df_original, target_feature, class_labels, cat_indicator = _make_dataset(
        problem_type, n=10
    )
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
    assert isinstance(oml_task, expected_cls), (
        f"Expected {expected_cls}, got {type(oml_task)}"
    )
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
    X, y, categorical_indicator, attribute_names = oml_dataset.get_data(
        target=oml_task.target_name
    )
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
        split={
            r: {
                f: {0: (np.array(tr), np.array(te))}
                for f, (tr, te) in splits[r].items()
            }
            for r in splits
        },
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
