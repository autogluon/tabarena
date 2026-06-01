"""Tests for the BeyondArena (Data Foundry) metadata workflow.

These avoid the optional ``data-foundry`` dependency and any network access by
patching the reference-metadata loader / materialization helpers on the
``tabarena.benchmark.task.data_foundry`` package (the bundle imports them lazily
from there at call time).
"""

from __future__ import annotations

import pandas as pd
import pytest
from tabarena.benchmark.task.metadata import (
    BeyondArenaMetadataBundle,
    SplitMetadata,
    TabArenaTaskMetadata,
)
from tabarena.benchmark.task.user_task import UserTask


# ---------------------------------------------------------------------------
# UserTask: portable (path-free) vs explicit-path task_id_str
# ---------------------------------------------------------------------------


def test_standardized_task_id_str_has_no_path():
    """A task with no explicit cache path serializes without a path segment."""
    task_id_str = UserTask(task_name="ds/uuid").task_id_str
    assert task_id_str.split("|")[:3] == ["UserTask", str(UserTask(task_name="ds/uuid").task_id), "ds/uuid"]
    assert len(task_id_str.split("|")) == 3


def test_custom_path_task_id_str_includes_path_and_roundtrips():
    task = UserTask(task_name="ds/uuid", task_cache_path="/custom/cache")
    assert task.task_id_str.endswith("|/custom/cache")
    restored = UserTask.from_task_id_str(task.task_id_str)
    assert str(restored.task_cache_path) == str(task.task_cache_path) == "/custom/cache"


def test_portable_task_id_str_roundtrips_to_ambient_cache():
    task = UserTask(task_name="ds/uuid")
    restored = UserTask.from_task_id_str(task.task_id_str)
    # No baked path: resolves against the ambient OpenML cache on both sides.
    assert restored.task_cache_path == task.task_cache_path
    assert restored.task_id == task.task_id


# ---------------------------------------------------------------------------
# BeyondArenaMetadataBundle: reference loading + filtering + resolution
# ---------------------------------------------------------------------------


def _ref_task(
    *,
    dataset_name: str,
    problem_type: str = "binary",
    uri: str | None = None,
) -> TabArenaTaskMetadata:
    uri = uri if uri is not None else f"{dataset_name}/uuid-{dataset_name}"
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
        problem_type=problem_type,
        is_classification=(problem_type != "regression"),
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
        task_id_str=f"UserTask|1|{uri}",
        data_foundry_uri=uri,
    )


def _reference_df() -> pd.DataFrame:
    tasks = [
        _ref_task(dataset_name="ds_bin", problem_type="binary"),
        _ref_task(dataset_name="ds_reg", problem_type="regression"),
    ]
    return pd.concat([t.to_dataframe() for t in tasks], ignore_index=True)


@pytest.fixture
def patched_data_foundry(monkeypatch):
    """Patch the lazily-imported data_foundry helpers; record materialize calls."""
    import tabarena.benchmark.task.data_foundry as df_pkg

    class _DummyCollection:
        name = "BeyondArena"

    monkeypatch.setattr(df_pkg, "get_beyond_arena_collection", lambda: _DummyCollection())
    monkeypatch.setattr(
        df_pkg,
        "load_reference_metadata",
        lambda **_: _reference_df(),
    )

    materialized: list[str] = []

    def _fake_materialize(*, collection, task_id_str, data_foundry_uri, **_):  # noqa: ARG001
        materialized.append(data_foundry_uri)
        return f"materialized::{data_foundry_uri}"

    monkeypatch.setattr(df_pkg, "materialize_task", _fake_materialize)
    return materialized


def test_inspection_without_materialize_does_not_download(patched_data_foundry):
    result = BeyondArenaMetadataBundle(materialize=False).load_task_metadata()

    assert {m.dataset_name for m in result} == {"ds_bin", "ds_reg"}
    assert patched_data_foundry == []  # nothing materialized / downloaded
    for m in result:
        # Reference ids are already portable (path-free) — left untouched.
        assert m.task_id_str == f"UserTask|1|{m.data_foundry_uri}"


def test_problem_type_filter_then_materialize(patched_data_foundry):
    result = BeyondArenaMetadataBundle(
        materialize=True,
        problem_types_to_run=["binary"],
    ).load_task_metadata()

    assert [m.dataset_name for m in result] == ["ds_bin"]
    # Only the surviving task is materialized, by its data_foundry_uri.
    assert patched_data_foundry == ["ds_bin/uuid-ds_bin"]
    assert result[0].task_id_str == "materialized::ds_bin/uuid-ds_bin"


def test_dataset_names_filter(patched_data_foundry):
    result = BeyondArenaMetadataBundle(
        materialize=False,
        dataset_names_to_run=["ds_reg"],
    ).load_task_metadata()

    assert [m.dataset_name for m in result] == ["ds_reg"]


def test_custom_metadata_passthrough_skips_reference_and_materialization(patched_data_foundry):
    """Passing custom metadata bypasses the reference loader; non-DF tasks aren't materialized."""
    custom = _ref_task(dataset_name="my_custom")
    custom.data_foundry_uri = None  # a user-supplied task, not from data_foundry

    result = BeyondArenaMetadataBundle(task_metadata=[custom], materialize=True).load_task_metadata()

    assert [m.dataset_name for m in result] == ["my_custom"]  # reference table was NOT loaded
    assert patched_data_foundry == []  # nothing to materialize without a data_foundry_uri
