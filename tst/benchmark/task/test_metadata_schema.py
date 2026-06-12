"""Tests for the TabArenaTaskMetadata schema round-trip."""

from __future__ import annotations

import io

import pandas as pd

from tabarena.benchmark.task.metadata import SplitMetadata, TabArenaTaskMetadata


def _task() -> TabArenaTaskMetadata:
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
        dataset_name="dummy",
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
        tabarena_task_name="Task-dummy",
        task_id_str="UserTask|1|dummy",
    )


def test_from_row_round_trips_none_fields_through_dataframe():
    """`to_dataframe` serializes None as NaN; `from_row` must map it back to None.

    A NaN leaking through (e.g. `group_on=nan`) reaches the validation-split logic
    as a truthy non-None value and crashes group-wise splitting (`KeyError: nan`).
    """
    task = _task()
    # The actual CSV round-trip every committed preset goes through: None -> NaN.
    buf = io.StringIO()
    task.to_dataframe().to_csv(buf, index=False)
    buf.seek(0)
    df = pd.read_csv(buf)
    assert pd.isna(df.loc[0, "group_on"])

    restored = TabArenaTaskMetadata.from_row(df.iloc[0])

    for field_name in (
        "stratify_on",
        "time_on",
        "group_on",
        "group_time_on",
        "group_labels",
        "split_time_horizon",
        "split_time_horizon_unit",
    ):
        assert getattr(restored, field_name) is None, field_name

    validation_metadata = restored.to_validation_metadata()
    assert validation_metadata.group_on is None
    assert validation_metadata.time_on is None


def test_from_row_keeps_set_optional_fields():
    task = _task()
    task.group_on = "patient_id"
    task.group_labels = "per_sample"
    df = task.to_dataframe()

    restored = TabArenaTaskMetadata.from_row(df.iloc[0])

    assert restored.group_on == "patient_id"
    assert restored.group_labels == "per_sample"
    assert restored.stratify_on is None
