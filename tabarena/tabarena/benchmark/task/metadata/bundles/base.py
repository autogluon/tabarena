"""Task (dataset x split) selection and metadata loading/filtering."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import pandas as pd

from tabarena.benchmark.task.metadata.schema import SplitMetadata, TabArenaTaskMetadata
from tabarena.benchmark.task.metadata.sources import TaskMetadataSource, resolve_source


@dataclass
class TabArenaMetadataBundle:
    """Encapsulates the source of task metadata and any filters applied on top of it.

    Can be used to define which tasks (datasets x splits) to run in a benchmark.
    """

    task_metadata: (
        TaskMetadataSource
        | Literal["TabArena-v0.1", "tabarena-v0.1", "BeyondArena"]
        | pd.DataFrame
        | list[TabArenaTaskMetadata]
        | str
        | Path
    )
    """Where the tasks come from, resolved via
    :func:`~tabarena.benchmark.task.metadata.sources.resolve_source`.

    Accepts a :class:`TaskMetadataSource` instance (e.g.
    ``DataFoundryTaskMetadataSource(my_collection)``), a registered suite literal
    (``"TabArena-v0.1"``, ``"BeyondArena"``), a pandas DataFrame, a list of
    TabArenaTaskMetadata, or a str/Path to a CSV file. DataFrames/CSVs are parsed
    per row via :meth:`TabArenaTaskMetadata.from_row`.
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
    materialize: bool = True
    """Whether to make the surviving (filtered) tasks runnable via the source's
    :meth:`~tabarena.benchmark.task.metadata.sources.base.TaskMetadataSource.materialize`
    hook. For sources backed by a remote collection (e.g. Data Foundry) this
    downloads + converts only the tasks that survived filtering; for already-local
    sources it is a no-op. Set False to inspect / filter metadata without any
    downloads.
    """

    def load_task_metadata(self) -> list[TabArenaTaskMetadata]:
        """Load, filter, and (optionally) materialize the task metadata to run."""
        source = resolve_source(self.task_metadata)
        task_metadata = source.load()

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

        if self.materialize:
            source.materialize(task_metadata)

        self._sanity_check_task_ids(task_metadata)
        self._print_filter_history(filter_history)
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