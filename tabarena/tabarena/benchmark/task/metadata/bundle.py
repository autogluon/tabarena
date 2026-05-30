"""Task (dataset x split) selection and metadata loading/filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from tabarena.benchmark.task.metadata.schema import SplitMetadata, TabArenaTaskMetadata


@dataclass
class TabArenaMetadataBundle:
    """Encapsulates the source of task metadata and any filters applied on top of it.

    Can be used to define which tasks (datasets x splits) to run in a benchmark.
    """

    task_metadata: Literal["tabarena-v0.1"] | pd.DataFrame | list[TabArenaTaskMetadata] | str | Path
    """Metadata that defines the tasks to benchmark.

    Accepts the `"tabarena-v0.1"` literal, a pandas DataFrame, a list of
    TabArenaTaskMetadata, or a str/Path to a CSV file (loaded as a DataFrame).
    We assume any DataFrame is created from a TabArenaTaskMetadata (or has all
    columns needed to parse each row via TabArenaTaskMetadata.from_row).
    """
    problem_types_to_run: list[str] = field(
        default_factory=lambda: [
            "binary",
            "multiclass",
            "regression",
        ]
    )
    """Problem types to run in the benchmark. Adjust as needed to run only
     specific problem types.
     Options: "binary", "regression", "multiclass".

    Can be understood as a filter on top of the TabArenaTaskMetadata.
    """
    split_indices_to_run: list[str] | Literal["lite"] | None = None
    """Split indices to run in the benchmark. Adjust as needed to run only specific
    splits. If None, we run all splits. If "lite", we run only the first split."""
    required_dtypes_to_run: list[str] | None = None
    """Adjust as needed to run only datasets with at least one column of data types.
    Options: "numeric", "categorical", "text", "datetime".
    If None, we do not require any data types.
    """
    forbidden_dtypes_to_run: list[str] | None = None
    """Adjust as needed to run only datasets without any columns of data types.
    Options: "numeric", "categorical", "text", "datetime".
    If None, we do not forbid any data types.
    """
    n_train_samples_to_run: tuple[int | None, int | None] | None = None
    """Tuple of lower and upper limit for the number of training samples of datasets run in the benchmark.
    Adjust as needed to run only datasets with a certain number of training samples.
    If None, we run all datasets.
    Lower limit is exclusive, upper limit is inclusive. For example, (0, 1000) runs only datasets with less
    than 1000 training samples. If a tuple value is None, there is no limit in that direction.
    """
    dataset_names_to_run: list[str] | None = None
    """List of dataset names to run in the benchmark. Adjust as needed to run only specific datasets.
    If None, we run all datasets. Matches against `dataset_name` of the task metadata.
    """

    def load_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Parse and filter the task metadata for jobs we want to run."""
        task_metadata = self._parse_task_metadata()

        # Unify format to be unrolled (one entry per split).
        task_metadata = [single_ttm for ttm in task_metadata for single_ttm in ttm.unroll_splits()]

        filter_steps = [
            ("problem types", self._filter_by_problem_types),
            ("splits", self._filter_by_split_indices),
            ("dataset names", self._filter_by_dataset_names),
            ("dtypes", self._filter_by_dtypes),
            ("dataset size", self._filter_by_train_samples),
        ]

        filter_history: list[tuple[str, int]] = [("Starting", len(task_metadata))]
        for label, filter_fn in filter_steps:
            task_metadata = filter_fn(task_metadata)
            filter_history.append((f"Filter to {label}", len(task_metadata)))

        self._sanity_check_task_ids(task_metadata)
        self._print_filter_history(filter_history)
        return task_metadata

    def _parse_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Resolve the configured `task_metadata` into a list of TabArenaTaskMetadata."""
        task_metadata = self.task_metadata

        if isinstance(task_metadata, str) and (task_metadata == "tabarena-v0.1"):
            task_metadata = self._load_tabarena_v0_1_task_metadata()
        if isinstance(task_metadata, (str, Path)):
            print(f"Loading task metadata from {task_metadata}...")
            task_metadata = pd.read_csv(task_metadata, index_col=False)
        if isinstance(task_metadata, pd.DataFrame):
            task_metadata = [TabArenaTaskMetadata.from_row(row) for _, row in task_metadata.iterrows()]
        assert all(isinstance(x, TabArenaTaskMetadata) for x in task_metadata)
        return task_metadata

    def _filter_by_problem_types(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        return [ttm for ttm in task_metadata if ttm.problem_type in self.problem_types_to_run]

    def _filter_by_split_indices(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.split_indices_to_run is None:
            return task_metadata

        if self.split_indices_to_run == "lite":
            split_indices_to_run = [SplitMetadata.get_split_index(repeat_i=0, fold_i=0)]
        else:
            split_indices_to_run = self.split_indices_to_run

        split_index_pattern = re.compile(r"^r\d+f\d+$")
        for split_index in split_indices_to_run:
            assert split_index_pattern.match(split_index), (
                f"Invalid SplitIndex format: {split_index!r}, expected 'r{{int}}f{{int}}'"
            )

        return [ttm for ttm in task_metadata if ttm.split_index in split_indices_to_run]

    def _filter_by_dataset_names(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.dataset_names_to_run is None:
            return task_metadata

        requested = set(self.dataset_names_to_run)
        available = {ttm.dataset_name for ttm in task_metadata}
        missing = requested - available
        if missing:
            raise ValueError(
                f"Requested dataset names not found in task metadata: {sorted(missing)}. "
                f"Available dataset names: {sorted(available)}"
            )
        return [ttm for ttm in task_metadata if ttm.dataset_name in requested]

    def _filter_by_dtypes(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if (self.forbidden_dtypes_to_run is None) and (self.required_dtypes_to_run is None):
            return task_metadata
        return [
            ttm
            for ttm in task_metadata
            if ttm.has_supported_dtypes(
                required_dtypes=self.required_dtypes_to_run,
                forbidden_dtypes=self.forbidden_dtypes_to_run,
            )
        ]

    def _filter_by_train_samples(self, task_metadata: list[TabArenaTaskMetadata]) -> list[TabArenaTaskMetadata]:
        if self.n_train_samples_to_run is None:
            return task_metadata

        lb, ub = self.n_train_samples_to_run
        lb = lb if lb is not None else 0
        ub = ub if ub is not None else float("inf")
        return [ttm for ttm in task_metadata if lb < ttm.splits_metadata[ttm.split_index].num_instances_train <= ub]

    @staticmethod
    def _sanity_check_task_ids(task_metadata: list[TabArenaTaskMetadata]) -> None:
        for ttm in task_metadata:
            if ttm.task_id_str is None:
                raise ValueError(f"Task metadata for task {ttm.tabarena_task_name} does not have a task_id_str!")

    @staticmethod
    def _print_filter_history(filter_history: list[tuple[str, int]]) -> None:
        lines = [
            f"Found {filter_history[-1][1]} tasks to run.",
            "\tTask Filter History:",
            *(f"\t({i}) {label}: {count}." for i, (label, count) in enumerate(filter_history, start=1)),
        ]
        print("\n".join(lines))

    @staticmethod
    def _load_tabarena_v0_1_task_metadata() -> list[TabArenaTaskMetadata]:
        """Load TabArena v0.1 task metadata and convert it to the new
        TabArenaTaskMetadata format (one entry per task, with splits unrolled).
        """
        print("Loading task metadata from TabArena v0.1 and converting to new TabArenaTaskMetadata format...")
        from tabarena.nips2025_utils.fetch_metadata import (
            load_curated_task_metadata,
        )

        metric_map = {
            "binary": "roc_auc",
            "multiclass": "log_loss",
            "regression": "rmse",
        }

        metadata = load_curated_task_metadata()
        task_metadata: list[TabArenaTaskMetadata] = []
        for row in metadata.itertuples():
            num_classes = row.num_classes
            num_instances = row.num_instances
            num_features = row.num_features

            n_repeats = row.tabarena_num_repeats
            n_folds = row.num_folds

            eval_metric = metric_map[row.problem_type]

            for repeat_i in range(n_repeats):
                for fold_i in range(n_folds):
                    split_index = SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i)
                    splits_metadata = {
                        split_index: SplitMetadata(
                            repeat=repeat_i,
                            fold=fold_i,
                            num_instances_train=num_instances * 2 / 3,
                            num_instances_test=num_instances * 1 / 3,
                            num_instance_groups_train=num_instances * 2 / 3,
                            num_instance_groups_test=num_instances * 1 / 3,
                            num_classes_train=num_classes,
                            num_classes_test=num_classes,
                            num_features_train=num_features,
                            num_features_test=num_features,
                        )
                    }

                    task_metadata.append(
                        TabArenaTaskMetadata(
                            task_id_str=row.task_id,
                            dataset_name=row.dataset_name,
                            tabarena_task_name=row.dataset_name,
                            problem_type=row.problem_type,
                            is_classification=row.is_classification,
                            target_name=row.target_feature,
                            stratify_on=row.target_feature if row.is_classification else None,
                            split_time_horizon=None,
                            split_time_horizon_unit=None,
                            time_on=None,
                            group_on=None,
                            group_time_on=None,
                            group_labels=None,
                            multiclass_max_n_classes_over_splits=num_classes,
                            multiclass_min_n_classes_over_splits=num_classes,
                            class_consistency_over_splits=True,
                            num_instances=num_instances,
                            num_features=num_features,
                            num_instance_groups=num_instances,
                            num_classes=num_classes,
                            splits_metadata=splits_metadata,
                            eval_metric=eval_metric,
                        )
                    )
        return task_metadata


@dataclass
class TabArenaV0pt1MetadataBundle(TabArenaMetadataBundle):
    """Metadata for full TabArena v0.1 benchmark: 51 datasets, 816 tasks."""
    task_metadata: str = "tabarena-v0.1"

@dataclass
class TabArenaV0pt1LiteMetadataBundle(TabArenaV0pt1MetadataBundle):
    """TabArena v0.1 Lite (first split of each dataset): 51 datasets, 51 tasks."""
    split_indices_to_run = "lite"