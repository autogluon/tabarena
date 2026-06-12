from __future__ import annotations

from typing import TYPE_CHECKING

from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLRegressionTask,
)

from tabarena.benchmark.task.metadata.compute import (
    compute_task_metadata,
    get_dataset_stats,
    get_num_instance_groups,
)

if TYPE_CHECKING:
    import pandas as pd

    from tabarena.benchmark.task.metadata import (
        GroupLabelTypes,
        SplitTimeHorizonTypes,
        SplitTimeHorizonUnitTypes,
        TabArenaTaskMetadata,
    )


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
        return get_num_instance_groups(X=X, group_on=group_on, group_labels=group_labels)

    def _get_dataset_stats(
        self,
        *,
        oml_dataset: pd.DataFrame,
        is_classification: bool,
        target_name: str,
    ) -> tuple[int, int, int, int]:
        return get_dataset_stats(
            dataset=oml_dataset,
            is_classification=is_classification,
            target_name=target_name,
            group_on=self.group_on,
            group_labels=self.group_labels,
        )

    def compute_metadata(
        self: TabArenaOpenMLSupervisedTask,
        tabarena_task_name: str | None = None,
        task_id_str: str | None = None,
    ) -> TabArenaTaskMetadata:
        """Get the metadata for the tasks.

        Thin projection of the task onto :func:`~tabarena.benchmark.task.metadata.compute.compute_task_metadata`
        (the one implementation of the task -> metadata mapping): the full dataset
        frame, the outer splits, and this task's split configuration.
        """
        oml_dataset_object = self.get_dataset()
        oml_dataset, *_ = oml_dataset_object.get_data()

        splits: dict[int, dict[int, tuple]] = {}
        for repeat_i, repeat_splits in self.split.split.items():
            splits[repeat_i] = {}
            for fold_i, samples_for_split in repeat_splits.items():
                assert len(samples_for_split) == 1, "Only one sample per split is supported so far!."
                train_idx, test_idx = samples_for_split[0]
                splits[repeat_i][fold_i] = (train_idx, test_idx)

        self._task_metadata = compute_task_metadata(
            dataset=oml_dataset,
            dataset_name=oml_dataset_object.name,
            target_name=self.target_name,
            is_classification=self.task_type_id.value == 1,
            eval_metric=self.evaluation_measure,
            splits=splits,
            stratify_on=self.stratify_on,
            group_on=self.group_on,
            time_on=self.time_on,
            group_time_on=self.group_time_on,
            group_labels=self.group_labels,
            split_time_horizon=self.split_time_horizon,
            split_time_horizon_unit=self.split_time_horizon_unit,
            tabarena_task_name=tabarena_task_name,
            task_id_str=task_id_str,
        )

        return self._task_metadata


class TabArenaOpenMLClassificationTask(TabArenaTaskMetadataMixin, OpenMLClassificationTask):
    """A local OpenMLClassificationTask with additional metadata for TabArena."""


class TabArenaOpenMLRegressionTask(TabArenaTaskMetadataMixin, OpenMLRegressionTask):
    """A local OpenMLRegressionTask with additional metadata for TabArena."""


# For typing
TabArenaOpenMLSupervisedTask = TabArenaOpenMLClassificationTask | TabArenaOpenMLRegressionTask
