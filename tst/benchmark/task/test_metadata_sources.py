"""Tests for the task-metadata source layer (load + resolve + materialize).

These avoid the optional ``data-foundry`` dependency and any network access by
patching the helpers on the ``tabarena.benchmark.task.data_foundry`` package that
:class:`DataFoundryTaskMetadataSource` imports lazily.
"""

from __future__ import annotations

import pandas as pd
import pytest
from tabarena.benchmark.task.metadata import (
    DataFoundryTaskMetadataSource,
    InMemoryTaskMetadataSource,
    SplitMetadata,
    TabArenaTaskMetadata,
    TabArenaV0pt1TaskMetadataSource,
    resolve_source,
)
from tabarena.benchmark.task.metadata.sources import tabarena_v0pt1 as v0pt1_mod
from tabarena.benchmark.task.metadata.sources.base import TaskMetadataSource


def _task(*, dataset_name: str, uri: str | None) -> TabArenaTaskMetadata:
    sm = SplitMetadata(
        repeat=0,
        fold=0,
        num_instances_train=80,
        num_instances_test=20,
        num_instance_groups_train=80,
        num_instance_groups_test=20,
        num_classes_train=2,
        num_classes_test=2,
        num_features_train=5,
        num_features_test=5,
    )
    return TabArenaTaskMetadata(
        dataset_name=dataset_name,
        problem_type="binary",
        is_classification=True,
        target_name="target",
        eval_metric="roc_auc",
        splits_metadata={sm.split_index: sm},
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
        tabarena_task_name=f"Task-{dataset_name}",
        task_id_str=f"UserTask|1|{uri or dataset_name}",
        data_foundry_uri=uri,
    )


# ---------------------------------------------------------------------------
# resolve_source dispatch
# ---------------------------------------------------------------------------


def test_resolve_source_passes_through_source_instances():
    src = TabArenaV0pt1TaskMetadataSource()
    assert resolve_source(src) is src


def test_resolve_source_maps_registered_literal():
    assert isinstance(resolve_source("TabArena-v0.1"), TabArenaV0pt1TaskMetadataSource)


# ---------------------------------------------------------------------------
# TabArenaV0pt1TaskMetadataSource — committed CSV + rebuild fallback
# ---------------------------------------------------------------------------


def test_v0pt1_source_loads_committed_csv():
    """The committed reference CSV is shipped and parses into task metadata."""
    assert v0pt1_mod.committed_metadata_path().exists()
    tasks = TabArenaV0pt1TaskMetadataSource().load()
    assert len(tasks) > 0
    assert all(isinstance(t, TabArenaTaskMetadata) for t in tasks)


def test_v0pt1_source_falls_back_to_rebuild_when_csv_missing(monkeypatch, tmp_path):
    monkeypatch.setattr(v0pt1_mod, "committed_metadata_path", lambda: tmp_path / "missing.csv")
    fake_curated = pd.DataFrame(
        [
            {
                "dataset_name": "fb_ds",
                "problem_type": "binary",
                "is_classification": True,
                "target_feature": "t",
                "task_id": "7",
                "num_instances": 50,
                "num_features": 3,
                "num_classes": 2,
                "tabarena_num_repeats": 1,
                "num_folds": 1,
            }
        ]
    )
    monkeypatch.setattr(
        "tabarena.nips2025_utils.fetch_metadata.load_curated_task_metadata",
        lambda: fake_curated,
    )

    tasks = TabArenaV0pt1TaskMetadataSource().load()
    assert [t.dataset_name for t in tasks] == ["fb_ds"]


def test_v0pt1_source_materialize_caches_unique_openml_tasks(monkeypatch):
    import openml

    fetched: list[int] = []
    monkeypatch.setattr(openml.tasks, "get_task", lambda task_id, **_: fetched.append(task_id))

    t1, t2, t3 = _task(dataset_name="a", uri=None), _task(dataset_name="b", uri=None), _task(dataset_name="c", uri=None)
    t1.task_id_str, t2.task_id_str, t3.task_id_str = "363", "363", "364"  # t1/t2 share a task id

    TabArenaV0pt1TaskMetadataSource().materialize([t1, t2, t3])

    assert fetched == [363, 364]  # de-duplicated, integer ids


def test_openml_source_honors_custom_cache_dir(monkeypatch, tmp_path):
    import openml

    set_dirs: list[str] = []
    monkeypatch.setattr(
        openml.config, "set_root_cache_directory", lambda root_cache_directory: set_dirs.append(root_cache_directory)
    )
    monkeypatch.setattr(openml.config, "get_cache_directory", lambda: str(tmp_path))
    monkeypatch.setattr(openml.tasks, "get_task", lambda task_id, **_: None)

    t = _task(dataset_name="a", uri=None)
    t.task_id_str = "5"
    TabArenaV0pt1TaskMetadataSource(openml_cache_dir=tmp_path / "custom").materialize([t])

    assert set_dirs == [str(tmp_path / "custom")]  # custom dir applied before caching


@pytest.mark.parametrize("data", [[], pd.DataFrame()], ids=["list", "dataframe"])
def test_resolve_source_wraps_in_memory(data):
    assert isinstance(resolve_source(data), InMemoryTaskMetadataSource)


# ---------------------------------------------------------------------------
# InMemoryTaskMetadataSource
# ---------------------------------------------------------------------------


def test_in_memory_source_roundtrips_dataframe():
    task = _task(dataset_name="ds", uri=None)
    loaded = InMemoryTaskMetadataSource(task.to_dataframe()).load()
    assert len(loaded) == 1
    assert loaded[0].dataset_name == "ds"


def test_in_memory_source_materialize_is_noop():
    task = _task(dataset_name="ds", uri="ds/uuid")
    # data_foundry_uri present, but in-memory source must not try to download.
    InMemoryTaskMetadataSource([task]).materialize([task])  # no error, no change
    assert task.task_id_str == "UserTask|1|ds/uuid"


# ---------------------------------------------------------------------------
# DataFoundryTaskMetadataSource — general purpose (any collection)
# ---------------------------------------------------------------------------


class _DummyCollection:
    name = "MyCollection"


@pytest.fixture
def patched_data_foundry(monkeypatch):
    import tabarena.benchmark.task.data_foundry as df_pkg

    ref_df = pd.concat(
        [
            _task(dataset_name="ds_a", uri="ds_a/uuid-a").to_dataframe(),
            _task(dataset_name="ds_b", uri=None).to_dataframe(),  # not data_foundry-backed
        ],
        ignore_index=True,
    )
    monkeypatch.setattr(df_pkg, "load_reference_metadata", lambda **_: ref_df)

    materialized: list[str] = []

    def _fake_materialize(*, collection, task_id_str, data_foundry_uri, **_):  # noqa: ARG001
        materialized.append(data_foundry_uri)
        return f"materialized::{data_foundry_uri}"

    monkeypatch.setattr(df_pkg, "materialize_task", _fake_materialize)
    return materialized


def test_data_foundry_source_load_uses_collection(patched_data_foundry):
    src = DataFoundryTaskMetadataSource(_DummyCollection())
    loaded = src.load()
    assert {t.dataset_name for t in loaded} == {"ds_a", "ds_b"}


def test_data_foundry_source_materialize_skips_tasks_without_uri(patched_data_foundry):
    src = DataFoundryTaskMetadataSource(_DummyCollection())
    tasks = src.load()
    src.materialize(tasks)

    # Only ds_a (which has a data_foundry_uri) is materialized.
    assert patched_data_foundry == ["ds_a/uuid-a"]
    by_name = {t.dataset_name: t for t in tasks}
    assert by_name["ds_a"].task_id_str == "materialized::ds_a/uuid-a"
    assert by_name["ds_b"].task_id_str == "UserTask|1|ds_b"  # untouched


def test_data_foundry_source_materialize_dedups_splits_of_same_dataset(patched_data_foundry):
    """A dataset unrolled into N splits is downloaded once, but every split is updated."""
    src = DataFoundryTaskMetadataSource(_DummyCollection())
    # Three splits of the same dataset (same data_foundry_uri), as produced by unroll_splits().
    splits = [_task(dataset_name="ds_a", uri="ds_a/uuid-a") for _ in range(3)]
    src.materialize(splits)

    # materialize_task is invoked once for the unique dataset, not once per split.
    assert patched_data_foundry == ["ds_a/uuid-a"]
    # ...yet the resolved task_id_str is propagated to all of the dataset's splits.
    assert all(s.task_id_str == "materialized::ds_a/uuid-a" for s in splits)
