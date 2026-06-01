from __future__ import annotations

import pandas as pd
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)

from tabarena.benchmark.task.metadata import (
    GroupLabelTypes,
    SplitMetadata,
    SplitTimeHorizonTypes,
    SplitTimeHorizonUnitTypes,
    TabArenaTaskMetadata,
)
from tabarena.benchmark.task.metadata.schema import derive_task_type


def _detect_binary_columns(feature_df: pd.DataFrame, *, sample_size: int = 10_000) -> set[str]:
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


class TabArenaTaskMetadataMixin:
    """A mixin class to add TabArena-specific metadata to OpenML tasks."""

    _task_metadata: TabArenaTaskMetadata | None = None

    # TODO: move split metadata to the split object itself and create a TabArena split object
    def __init__(
        self,
        *,
        stratify_on: str | None = None,
        time_on: str | None = None,
        group_on: str | list[str] | None = None,
        group_time_on: str | None = None,
        group_labels: GroupLabelTypes | None = None,
        split_time_horizon: SplitTimeHorizonTypes | None = None,
        split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
        **kwargs,
    ) -> None:
        """Checkout Data Foundry's PredictiveMLTaskMetadata for more information."""
        super().__init__(**kwargs)
        self.stratify_on = stratify_on
        self.group_on = group_on
        self.time_on = time_on
        self.group_time_on = group_time_on
        self.group_labels = group_labels
        self._task_metadata = None
        self.split_time_horizon = split_time_horizon
        self.split_time_horizon_unit = split_time_horizon_unit

    @staticmethod
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

    def _get_dataset_stats(
        self, *, oml_dataset: pd.DataFrame, is_classification: bool, target_name: str
    ) -> tuple[int, int, int, int]:
        num_instance = len(oml_dataset)
        num_features = oml_dataset.shape[1] - 1  # -1 for target
        num_classes = -1
        if is_classification:
            num_classes = max(
                int(oml_dataset[target_name].nunique()),
                int(oml_dataset[target_name].nunique()),
            )

        num_instance_groups = num_instance

        # Resolve instance groups
        if self.group_on is not None:
            num_instance_groups = self.get_num_instance_groups(
                X=oml_dataset,
                group_on=self.group_on,
                group_labels=self.group_labels,
            )

        return (
            num_instance,
            num_features,
            num_classes,
            num_instance_groups,
        )

    def compute_metadata(
        self: TabArenaOpenMLSupervisedTask,
        tabarena_task_name: str | None = None,
        task_id_str: str | None = None,
    ) -> TabArenaTaskMetadata:
        """Get the metadata for the tasks."""
        oml_dataset_object = self.get_dataset()
        oml_dataset, *_ = oml_dataset_object.get_data()
        dataset_name = oml_dataset_object.name
        eval_metric = self.evaluation_measure
        is_classification = self.task_type_id.value == 1
        target_name = self.target_name

        task_problem_type = None
        num_classes_list = []

        # Get overall stats of the dataset
        (
            full_num_instance,
            full_num_features,
            full_num_classes,
            full_num_instance_groups,
        ) = self._get_dataset_stats(
            oml_dataset=oml_dataset,
            is_classification=is_classification,
            target_name=target_name,
        )

        splits_metadata = {}
        for repeat_i, splits in self.split.split.items():
            for fold_i, samples_for_split in splits.items():
                assert len(samples_for_split) == 1, "Only one sample per split is supported so far!."
                train_idx, test_idx = samples_for_split[0]

                (
                    train_num_instance,
                    train_num_features,
                    train_num_classes,
                    train_num_instance_groups,
                ) = self._get_dataset_stats(
                    oml_dataset=oml_dataset.iloc[train_idx],
                    is_classification=is_classification,
                    target_name=target_name,
                )
                (
                    test_num_instance,
                    test_num_features,
                    test_num_classes,
                    test_num_instance_groups,
                ) = self._get_dataset_stats(
                    oml_dataset=oml_dataset.iloc[test_idx],
                    is_classification=is_classification,
                    target_name=target_name,
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
        if self.group_on is not None:
            if isinstance(self.group_on, list):
                excluded_columns.update(self.group_on)
            else:
                excluded_columns.add(self.group_on)
        feature_df = oml_dataset.drop(columns=excluded_columns)

        # FIXME: make this less strict?
        if len(feature_df.select_dtypes(include=["object"]).columns) > 0:
            raise ValueError(
                "Object dtype columns are not supported. Please convert them to string dtype or categorical dtype!"
            )

        # Independent dtype flags
        binary_cols = _detect_binary_columns(feature_df)
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
        has_high_cardinality_categorical = any(
            feature_df[c].nunique(dropna=True) > 50
            for c in non_binary_categorical_cols
        )

        # Warehouse-level counts (consistent with the Data Foundry warehouse computation,
        # see data-foundry simple_metadata_exploration). Text/datetime counts are raw
        # (binary-inclusive); the after-preprocessing estimate expands text/datetime.
        category_only_cols = feature_df.select_dtypes(include=["category"]).columns
        num_text_cols = len(text_cols)
        num_datetime_cols = len(datetime_cols)
        num_high_cardinality_cats = int(
            sum(feature_df[c].nunique(dropna=True) > 50 for c in category_only_cols)
        )
        num_cols_after_preprocessing = (
            sum(c not in binary_cols for c in numerical_cols)
            + sum(c not in binary_cols for c in categorical_cols)
            + num_text_cols * 32
            + len(binary_cols)
            + num_datetime_cols * 10
        )
        missing_value_fraction = (
            float(feature_df.isna().to_numpy().sum() / feature_df.size) if feature_df.size else 0.0
        )

        self._task_metadata = TabArenaTaskMetadata(
            dataset_name=dataset_name,
            eval_metric=eval_metric,
            splits_metadata=splits_metadata,
            is_classification=is_classification,
            problem_type=task_problem_type,
            multiclass_min_n_classes_over_splits=min_n_classes,
            multiclass_max_n_classes_over_splits=max_n_classes,
            class_consistency_over_splits=class_consistency_over_splits,
            target_name=target_name,
            stratify_on=self.stratify_on,
            group_on=self.group_on,
            time_on=self.time_on,
            group_time_on=self.group_time_on,
            group_labels=self.group_labels,
            tabarena_task_name=tabarena_task_name,
            task_id_str=task_id_str,
            num_instances=full_num_instance,
            num_features=full_num_features,
            num_classes=full_num_classes,
            num_instance_groups=full_num_instance_groups,
            split_time_horizon=self.split_time_horizon,
            split_time_horizon_unit=self.split_time_horizon_unit,
            has_datetime=has_datetime,
            has_text=has_text,
            has_categorical=has_categorical,
            has_numerical=has_numerical,
            has_binary=has_binary,
            has_high_cardinality_categorical=has_high_cardinality_categorical,
            task_type=derive_task_type(time_on=self.time_on, group_on=self.group_on),
            num_text_cols=num_text_cols,
            num_high_cardinality_cats=num_high_cardinality_cats,
            num_cols_after_preprocessing=num_cols_after_preprocessing,
            missing_value_fraction=missing_value_fraction,
        )

        return self._task_metadata


class TabArenaOpenMLClassificationTask(TabArenaTaskMetadataMixin, OpenMLClassificationTask):
    """A local OpenMLClassificationTask with additional metadata for TabArena."""


class TabArenaOpenMLRegressionTask(TabArenaTaskMetadataMixin, OpenMLRegressionTask):
    """A local OpenMLRegressionTask with additional metadata for TabArena."""


# For typing
TabArenaOpenMLSupervisedTask = TabArenaOpenMLClassificationTask | TabArenaOpenMLRegressionTask
