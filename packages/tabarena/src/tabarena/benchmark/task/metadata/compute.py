"""Compute a task's exact ``TabArenaTaskMetadata`` from generic inputs.

:func:`compute_task_metadata` is the one implementation of the task -> metadata
mapping, operating on plain ``(dataset, target, splits, split-config)`` inputs —
no task object of any particular source required. Both task-side entry points
delegate here:

* ``TabArenaTaskMetadataMixin.compute_metadata`` (TabArena's local OpenML task
  classes), and
* ``TaskWrapper.compute_metadata`` (any loaded task wrapper, out of the box),

so "computed from the task" and "stored in a collection" can never drift
structurally — only stale stored values remain detectable (see
``TaskWrapper.validate_metadata``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.task.metadata.schema import (
    GroupLabelTypes,
    SplitMetadata,
    TabArenaTaskMetadata,
    derive_task_type,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from tabarena.benchmark.task.metadata.schema import (
        SplitTimeHorizonTypes,
        SplitTimeHorizonUnitTypes,
    )


def detect_binary_columns(feature_df: pd.DataFrame, *, sample_size: int = 10_000) -> set[str]:
    """Return columns of `feature_df` with exactly 2 distinct non-null values.

    Two-stage scan to avoid a full nunique() pass on every column for wide/large
    frames. We first check an evenly-spaced sample: nunique on a subset is a
    lower bound on the full nunique, so any column with >2 uniques in the sample
    cannot be binary and is skipped. Surviving candidates are verified on the
    full column.
    """
    n = len(feature_df)
    if n <= sample_size:
        return {c for c in feature_df.columns if feature_df[c].nunique(dropna=True) == 2}

    sample = feature_df.iloc[:: max(1, n // sample_size)]
    candidates = [c for c in feature_df.columns if sample[c].nunique(dropna=True) <= 2]
    return {c for c in candidates if feature_df[c].nunique(dropna=True) == 2}


def get_num_instance_groups(
    *,
    X: pd.DataFrame,
    group_on: str | list[str] | None,
    group_labels: GroupLabelTypes | None,
) -> int:
    """Compute the number of instance groups in data based on the group_on."""
    if (group_on is None) or (group_labels == GroupLabelTypes.PER_SAMPLE):
        return len(X)

    group_on = group_on if isinstance(group_on, list) else [group_on]
    return X.groupby(group_on, dropna=False, observed=True).ngroups


def get_dataset_stats(
    *,
    dataset: pd.DataFrame,
    is_classification: bool,
    target_name: str,
    group_on: str | list[str] | None = None,
    group_labels: GroupLabelTypes | None = None,
) -> tuple[int, int, int, int]:
    """Return ``(num_instances, num_features, num_classes, num_instance_groups)`` of one frame.

    ``dataset`` is the full frame *including* the target (and any group columns);
    ``num_classes`` is ``-1`` for regression.
    """
    num_instance = len(dataset)
    num_features = dataset.shape[1] - 1  # -1 for target
    num_classes = -1
    if is_classification:
        num_classes = int(dataset[target_name].nunique())

    num_instance_groups = num_instance

    # Resolve instance groups
    if group_on is not None:
        num_instance_groups = get_num_instance_groups(
            X=dataset,
            group_on=group_on,
            group_labels=group_labels,
        )

    return (
        num_instance,
        num_features,
        num_classes,
        num_instance_groups,
    )


def compute_task_metadata(
    *,
    dataset: pd.DataFrame,
    target_name: str,
    is_classification: bool,
    splits: dict[int, dict[int, tuple[Sequence[int], Sequence[int]]]],
    dataset_name: str | None = None,
    eval_metric: str | None = None,
    stratify_on: str | None = None,
    group_on: str | list[str] | None = None,
    time_on: str | None = None,
    group_time_on: str | None = None,
    group_labels: GroupLabelTypes | None = None,
    split_time_horizon: SplitTimeHorizonTypes | None = None,
    split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
    tabarena_task_name: str | None = None,
    task_id_str: str | None = None,
) -> TabArenaTaskMetadata:
    """Compute the exact ``TabArenaTaskMetadata`` of one task from generic inputs.

    Parameters
    ----------
    dataset: pd.DataFrame
        The full task frame *including* the target column (and any group/time
        columns). Dtypes must be resolved (numericals as numbers, categoricals as
        ``category``, text as ``string``, dates as datetime); ``object`` feature
        columns are rejected.
    target_name: str
        Name of the target column in ``dataset``.
    is_classification: bool
        Whether the task is a classification task (``num_classes`` is ``-1`` and the
        class-consistency fields ``None`` otherwise).
    splits: dict[int, dict[int, tuple[Sequence[int], Sequence[int]]]]
        The outer evaluation splits as ``{repeat: {fold: (train_idx, test_idx)}}``,
        with positional (0-based) row indices into ``dataset``.
    dataset_name, eval_metric:
        Recorded verbatim on the metadata (``eval_metric=None`` means the TabArena
        per-problem-type default applies downstream).
    stratify_on, group_on, time_on, group_time_on, group_labels:
        The split-configuration columns (see ``TabArenaTaskMetadata``).
    split_time_horizon, split_time_horizon_unit:
        The temporal-split horizon configuration.
    tabarena_task_name, task_id_str:
        The task's identity (results ``dataset`` key and serialized spec id).
    """
    task_problem_type = None
    num_classes_list = []

    # Get overall stats of the dataset
    (
        full_num_instance,
        full_num_features,
        full_num_classes,
        full_num_instance_groups,
    ) = get_dataset_stats(
        dataset=dataset,
        is_classification=is_classification,
        target_name=target_name,
        group_on=group_on,
        group_labels=group_labels,
    )

    splits_metadata = {}
    for repeat_i, repeat_splits in splits.items():
        for fold_i, (train_idx, test_idx) in repeat_splits.items():
            (
                train_num_instance,
                train_num_features,
                train_num_classes,
                train_num_instance_groups,
            ) = get_dataset_stats(
                dataset=dataset.iloc[train_idx],
                is_classification=is_classification,
                target_name=target_name,
                group_on=group_on,
                group_labels=group_labels,
            )
            (
                test_num_instance,
                test_num_features,
                test_num_classes,
                test_num_instance_groups,
            ) = get_dataset_stats(
                dataset=dataset.iloc[test_idx],
                is_classification=is_classification,
                target_name=target_name,
                group_on=group_on,
                group_labels=group_labels,
            )

            # Resolve problem type
            max_num_classes = max(train_num_classes, test_num_classes)
            if max_num_classes == -1:
                split_problem_type = "regression"
            elif max_num_classes == 2:
                split_problem_type = "binary"
                num_classes_list.append(max_num_classes)
            else:
                split_problem_type = "multiclass"
                num_classes_list.append(max_num_classes)
            if task_problem_type is None:
                task_problem_type = split_problem_type
            else:
                assert task_problem_type == split_problem_type, "All splits must have the same problem type."
            s_index = SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i)
            splits_metadata[s_index] = SplitMetadata(
                repeat=repeat_i,
                fold=fold_i,
                num_instances_train=train_num_instance,
                num_instances_test=test_num_instance,
                num_instance_groups_train=train_num_instance_groups,
                num_instance_groups_test=test_num_instance_groups,
                num_classes_train=train_num_classes,
                num_classes_test=test_num_classes,
                num_features_train=train_num_features,
                num_features_test=test_num_features,
            )

    if len(num_classes_list) == 0:
        min_n_classes = None
        max_n_classes = None
        class_consistency_over_splits = None
    else:
        min_n_classes = min(num_classes_list)
        max_n_classes = max(num_classes_list)
        class_consistency_over_splits = min_n_classes == max_n_classes

    # Detect feature dtype flags (exclude target column)
    excluded_columns = {target_name}
    if group_on is not None:
        if isinstance(group_on, list):
            excluded_columns.update(group_on)
        else:
            excluded_columns.add(group_on)
    feature_df = dataset.drop(columns=excluded_columns)

    # FIXME: make this less strict?
    if len(feature_df.select_dtypes(include=["object"]).columns) > 0:
        raise ValueError(
            "Object dtype columns are not supported. Please convert them to string dtype or categorical dtype!",
        )

    # Independent dtype flags
    binary_cols = detect_binary_columns(feature_df)
    numerical_cols = feature_df.select_dtypes(include=["number"], exclude=["bool"]).columns
    categorical_cols = feature_df.select_dtypes(include=["category", "bool"]).columns
    datetime_cols = list(feature_df.select_dtypes(include=["datetime", "datetimetz"]).columns)
    datetime_cols += [c for c in feature_df.columns if isinstance(feature_df[c].dtype, pd.PeriodDtype)]
    text_cols = feature_df.select_dtypes(include=["string"]).columns

    has_numerical = sum(c not in binary_cols for c in numerical_cols) > 0
    has_datetime = sum(c not in binary_cols for c in datetime_cols) > 0
    has_text = sum(c not in binary_cols for c in text_cols) > 0
    has_binary = len(binary_cols) > 0

    non_binary_categorical_cols = [c for c in categorical_cols if c not in binary_cols]
    has_categorical = len(non_binary_categorical_cols) > 0
    has_high_cardinality_categorical = any(feature_df[c].nunique(dropna=True) > 50 for c in non_binary_categorical_cols)

    # Warehouse-level counts (consistent with the Data Foundry warehouse computation,
    # see data-foundry simple_metadata_exploration). Text/datetime counts are raw
    # (binary-inclusive); the after-preprocessing estimate expands text/datetime.
    category_only_cols = feature_df.select_dtypes(include=["category"]).columns
    num_text_cols = len(text_cols)
    num_datetime_cols = len(datetime_cols)
    num_high_cardinality_cats = int(
        sum(feature_df[c].nunique(dropna=True) > 50 for c in category_only_cols),
    )
    num_cols_after_preprocessing = (
        sum(c not in binary_cols for c in numerical_cols)
        + sum(c not in binary_cols for c in categorical_cols)
        + num_text_cols * 32
        + len(binary_cols)
        + num_datetime_cols * 10
    )
    missing_value_fraction = float(feature_df.isna().to_numpy().sum() / feature_df.size) if feature_df.size else 0.0

    return TabArenaTaskMetadata(
        dataset_name=dataset_name,
        eval_metric=eval_metric,
        splits_metadata=splits_metadata,
        is_classification=is_classification,
        problem_type=task_problem_type,
        multiclass_min_n_classes_over_splits=min_n_classes,
        multiclass_max_n_classes_over_splits=max_n_classes,
        class_consistency_over_splits=class_consistency_over_splits,
        target_name=target_name,
        stratify_on=stratify_on,
        group_on=group_on,
        time_on=time_on,
        group_time_on=group_time_on,
        group_labels=group_labels,
        tabarena_task_name=tabarena_task_name,
        task_id_str=task_id_str,
        num_instances=full_num_instance,
        num_features=full_num_features,
        num_classes=full_num_classes,
        num_instance_groups=full_num_instance_groups,
        split_time_horizon=split_time_horizon,
        split_time_horizon_unit=split_time_horizon_unit,
        has_datetime=has_datetime,
        has_text=has_text,
        has_categorical=has_categorical,
        has_numerical=has_numerical,
        has_binary=has_binary,
        has_high_cardinality_categorical=has_high_cardinality_categorical,
        task_type=derive_task_type(time_on=time_on, group_on=group_on),
        num_text_cols=num_text_cols,
        num_high_cardinality_cats=num_high_cardinality_cats,
        num_cols_after_preprocessing=num_cols_after_preprocessing,
        missing_value_fraction=missing_value_fraction,
    )
