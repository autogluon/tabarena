"""InMemoryTaskWrapper + the native UserTask create/save/load path.

The native path must be a drop-in replacement for the legacy local-OpenML-task
fabrication: same metadata, same splits, same data — without any OpenML object.
Legacy caches on disk (pickled local OpenML tasks) must keep loading.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import StratifiedKFold

from tabarena.benchmark.task import InMemoryTaskWrapper, UserTask, from_sklearn_splits_to_user_task_splits
from tabarena.benchmark.task.openml import OpenMLTaskWrapper


def _toy_dataset(n: int = 60) -> pd.DataFrame:
    rng = np.random.RandomState(0)
    df = pd.DataFrame(rng.normal(size=(n, 4)), columns=[f"num_{i}" for i in range(4)])
    df["cat"] = pd.Categorical(["a", "b", "c"] * (n // 3))
    df["target"] = [0, 1] * (n // 2)
    return df


def _toy_splits(dataset: pd.DataFrame, n_splits: int = 3) -> dict:
    return from_sklearn_splits_to_user_task_splits(
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(
            dataset.drop(columns="target"),
            dataset["target"],
        ),
        n_splits=n_splits,
    )


def _native_task(tmp_path, name: str = "toy_native") -> tuple[UserTask, InMemoryTaskWrapper]:
    dataset = _toy_dataset()
    task = UserTask(task_name=name, task_cache_path=tmp_path)
    wrapper = task.create_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=_toy_splits(dataset),
    )
    return task, wrapper


def test_create_task_builds_in_memory_wrapper(tmp_path):
    task, wrapper = _native_task(tmp_path)
    assert isinstance(wrapper, InMemoryTaskWrapper)
    assert wrapper.problem_type == "binary"
    assert wrapper.label == "target"
    assert wrapper.task_id == task.task_id
    assert wrapper.get_split_dimensions() == (1, 3, 1)
    # Identity is filled in by create_task.
    assert wrapper.metadata.tabarena_task_name == task.tabarena_task_name
    assert wrapper.metadata.task_id_str == task.task_id_str
    assert wrapper.dataset_name == f"Dataset-{task.task_name}"

    X_train, y_train, X_test, y_test = wrapper.get_train_test_split(fold=0)
    assert len(X_train) + len(X_test) == 60
    assert set(X_train.columns) == {"num_0", "num_1", "num_2", "num_3", "cat"}
    assert y_train.name == "target"


def test_native_save_load_round_trip(tmp_path):
    task, created = _native_task(tmp_path)
    task.save_task(created)

    loaded = task.load()
    assert isinstance(loaded, InMemoryTaskWrapper)
    assert loaded.lazy_load_data  # data re-read from the cache file per access
    # The loaded task is exactly the created one: metadata and split data round-trip.
    loaded.validate_metadata(created.metadata)
    for fold in range(3):
        created_train, _, created_test, _ = created.get_train_test_split(fold=fold)
        loaded_train, _, loaded_test, _ = loaded.get_train_test_split(fold=fold)
        pd.testing.assert_frame_equal(created_train, loaded_train)
        pd.testing.assert_frame_equal(created_test, loaded_test)
    # Dtypes survive the round trip exactly (pickle, not parquet/CSV).
    assert loaded.get_X_y()[0]["cat"].dtype == "category"


def test_native_path_matches_legacy_path(tmp_path):
    """The native task is a drop-in for the legacy local-OpenML fabrication."""
    dataset = _toy_dataset()
    splits = _toy_splits(dataset)

    native_task = UserTask(task_name="toy_equiv", task_cache_path=tmp_path / "native")
    native = native_task.create_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    native_task.save_task(native)

    legacy_task = UserTask(task_name="toy_equiv", task_cache_path=tmp_path / "legacy")
    oml_task = legacy_task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=splits,
    )
    legacy_meta = oml_task.compute_metadata(
        tabarena_task_name=legacy_task.tabarena_task_name,
        task_id_str=legacy_task.task_id_str,
    )
    legacy_task.save_local_openml_task(oml_task)

    # Same metadata (the embedded explicit cache path differs by construction)...
    native_dict = {**native.metadata.to_dict(), "task_id_str": None}
    legacy_dict = {**legacy_meta.to_dict(), "task_id_str": None}
    assert native_dict == legacy_dict
    # ...and the same loaded data/splits through both wrappers.
    native_loaded = native_task.load()
    legacy_loaded = legacy_task.load()
    assert isinstance(legacy_loaded, OpenMLTaskWrapper)
    assert native_loaded.eval_metric == legacy_loaded.eval_metric
    for fold in range(3):
        n_train, n_y, n_test, _ = native_loaded.get_train_test_split(fold=fold)
        l_train, l_y, l_test, _ = legacy_loaded.get_train_test_split(fold=fold)
        pd.testing.assert_frame_equal(n_train, l_train)
        pd.testing.assert_frame_equal(n_test, l_test)
        pd.testing.assert_series_equal(n_y, l_y, check_dtype=False)


def test_legacy_cache_still_loads_via_load(tmp_path):
    """Existing on-disk caches (pickled local OpenML tasks) keep working."""
    dataset = _toy_dataset()
    task = UserTask(task_name="toy_legacy", task_cache_path=tmp_path)
    oml_task = task.create_local_openml_task(
        dataset=dataset,
        target_feature="target",
        problem_type="classification",
        splits=_toy_splits(dataset),
    )
    task.save_local_openml_task(oml_task)

    loaded = task.load()
    assert isinstance(loaded, OpenMLTaskWrapper)
    assert loaded.problem_type == "binary"
    X_train, *_ = loaded.get_train_test_split(fold=0)
    assert len(X_train) == 40


def test_load_local_openml_task_rejects_native_format(tmp_path):
    task, wrapper = _native_task(tmp_path)
    task.save_task(wrapper)
    with pytest.raises(TypeError, match="native TabArena format"):
        task.load_local_openml_task()


def test_load_missing_cache_raises(tmp_path):
    task = UserTask(task_name="missing", task_cache_path=tmp_path)
    with pytest.raises(FileNotFoundError):
        task.load()
