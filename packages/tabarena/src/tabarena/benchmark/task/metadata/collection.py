from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.task.metadata.schema import (
    SplitMetadata,
    TabArenaTaskMetadata,
    derive_task_type,
    to_legacy_task_metadata,
)

if TYPE_CHECKING:
    from collections.abc import Iterator


class TaskMetadataCollection:
    """A ``list[TabArenaTaskMetadata]`` plus the derived views its consumers need.

    The wrapped list is the source of truth; the views are derived on demand:

    * :meth:`dataset_names` / :meth:`dataset_fold_repeats` — native, built straight from
      the tasks (no DataFrame, no legacy columns).
    * :meth:`per_dataset_frame` — one row per dataset with the dataset-level metadata
      (native column names), for joining onto a results frame.
    * :meth:`dataset_to_tid` / :meth:`to_legacy_df` — the boundary adapters for consumers
      that are still keyed on the legacy schema / OpenML ``tid`` (tabrepo repo generation,
      ``ExperimentBatchRunner``). Keeping the legacy conversion behind one named method
      makes it the single chokepoint to migrate later.

    Mirrors :class:`~tabarena.models._method_metadata_collection.MethodMetadataCollection`
    on the methods axis. The wrapped list may be "unrolled" (one entry per split) or hold
    multi-split tasks; every view iterates ``splits_metadata`` so both forms behave the same.
    """

    def __init__(self, tasks: list[TabArenaTaskMetadata]):
        self._tasks = list(tasks)

    # ------------------------------------------------------------------ list-like
    @property
    def tasks(self) -> list[TabArenaTaskMetadata]:
        return self._tasks

    def __len__(self) -> int:
        return len(self._tasks)

    def __iter__(self) -> Iterator[TabArenaTaskMetadata]:
        return iter(self._tasks)

    def __getitem__(self, idx):
        return self._tasks[idx]

    # ------------------------------------------------------------------ native views
    def dataset_names(self) -> list[str]:
        """Unique dataset names, in first-seen order."""
        seen: dict[str, None] = {}
        for t in self._tasks:
            seen.setdefault(t.tabarena_task_name, None)
        return list(seen)

    def dataset_fold_repeats(self) -> list[tuple[str, int, int]]:
        """All ``(dataset, fold, repeat)`` triplets, from each task's ``splits_metadata``."""
        return [
            (t.tabarena_task_name, split.fold, split.repeat)
            for t in self._tasks
            for split in t.splits_metadata.values()
        ]

    def per_dataset_frame(self) -> pd.DataFrame:
        """One row per dataset with the dataset-level (split-invariant) metadata.

        Uses native column names (``problem_type``, ``num_features``, ``num_classes``, ...)
        plus a ``dataset`` key column. Stage 2 can extend this with predicate-derived
        columns (e.g. a ``max_train_rows`` aggregate over splits).
        """
        rows: dict[str, dict] = {}
        for t in self._tasks:
            # Static fields are dataset-level, so the first task seen per dataset wins.
            rows.setdefault(t.tabarena_task_name, t.to_dict(exclude_splits_metadata=True))
        frame = pd.DataFrame(list(rows.values()))
        if not frame.empty:
            frame["dataset"] = list(rows.keys())
        return frame

    # ------------------------------------------------------------------ legacy boundary
    @classmethod
    def from_legacy_df(cls, task_metadata: pd.DataFrame) -> TaskMetadataCollection:
        """Build a collection from a legacy ``task_metadata`` DataFrame (**lossy** shim).

        Inverse of :meth:`to_legacy_df`, for callers that still hand ``TabArenaContext`` a
        legacy frame (the ``load_task_metadata`` format: one row per dataset with
        ``tid``/``dataset``/``n_folds``/``n_repeats``/``n_features``/``n_classes``/
        ``n_samples_train_per_fold``/...). Prefer the native path
        (:meth:`TabArenaMetadataBundle.load_collection`) when you can — the legacy frame
        lacks most rich fields, so the rebuilt tasks are structurally valid but sparse:

        * derived: ``eval_metric`` (from ``problem_type``), ``is_classification``, and the
          ``splits_metadata`` grid (per-split sizes from ``n_samples_train/test_per_fold``).
        * preserved as-is: ``num_classes`` keeps the legacy encoding (``0`` for regression),
          not the schema's documented ``-1``.
        * unavailable -> ``None``: dtype flags, domain/year/source, group/time split fields,
          multiclass min/max, class-consistency.

        For a *native*-schema frame (``TabArenaTaskMetadata.to_dataframe`` format) use
        :meth:`TabArenaTaskMetadata.from_row` instead — that round-trips losslessly.
        """
        metric_map = {"binary": "roc_auc", "multiclass": "log_loss", "regression": "rmse"}
        required = [
            "problem_type",
            "n_folds",
            "n_repeats",
            "n_features",
            "n_classes",
            "NumberOfInstances",
            "tid",
            "n_samples_train_per_fold",
            "n_samples_test_per_fold",
        ]
        missing = [c for c in required if c not in task_metadata.columns]
        if missing:
            raise ValueError(f"Legacy task_metadata is missing required columns: {missing}")
        if "dataset" not in task_metadata.columns and "name" not in task_metadata.columns:
            raise ValueError("Legacy task_metadata must have a 'dataset' or 'name' column.")

        tasks: list[TabArenaTaskMetadata] = []
        for row in task_metadata.to_dict("records"):
            dataset = row.get("dataset") or row.get("name")
            problem_type = row["problem_type"]
            n_classes = int(row["n_classes"])
            num_features = int(row["n_features"])
            num_instances = int(row["NumberOfInstances"])
            n_train = int(row["n_samples_train_per_fold"])
            n_test = int(row["n_samples_test_per_fold"])
            n_folds, n_repeats = int(row["n_folds"]), int(row["n_repeats"])
            target_name = row.get("target_feature")
            if target_name is not None and pd.isna(target_name):
                target_name = None

            splits_metadata = {
                SplitMetadata.get_split_index(repeat_i=repeat_i, fold_i=fold_i): SplitMetadata(
                    repeat=repeat_i,
                    fold=fold_i,
                    num_instances_train=n_train,
                    num_instances_test=n_test,
                    num_instance_groups_train=n_train,
                    num_instance_groups_test=n_test,
                    num_classes_train=n_classes,
                    num_classes_test=n_classes,
                    num_features_train=num_features,
                    num_features_test=num_features,
                )
                for repeat_i in range(n_repeats)
                for fold_i in range(n_folds)
            }

            tasks.append(
                TabArenaTaskMetadata(
                    dataset_name=dataset,
                    tabarena_task_name=dataset,
                    problem_type=problem_type,
                    is_classification=problem_type != "regression",
                    target_name=target_name,
                    eval_metric=metric_map[problem_type],
                    splits_metadata=splits_metadata,
                    split_time_horizon=None,
                    split_time_horizon_unit=None,
                    stratify_on=None,
                    time_on=None,
                    group_on=None,
                    group_time_on=None,
                    group_labels=None,
                    multiclass_min_n_classes_over_splits=None,
                    multiclass_max_n_classes_over_splits=None,
                    class_consistency_over_splits=None,
                    num_instances=num_instances,
                    num_features=num_features,
                    num_classes=n_classes,
                    num_instance_groups=num_instances,
                    task_id_str=str(int(row["tid"])),
                    task_type=derive_task_type(time_on=None, group_on=None),
                ),
            )
        return cls(tasks)

    def to_legacy_df(self) -> pd.DataFrame:
        """The legacy one-row-per-dataset ``task_metadata`` DataFrame.

        The single named adapter for consumers that still require the legacy schema
        (tabrepo repo generation, ``ExperimentBatchRunner``). Delegates to
        :func:`~tabarena.benchmark.task.metadata.schema.to_legacy_task_metadata`.
        """
        return to_legacy_task_metadata(self._tasks)

    def dataset_to_tid(self) -> dict[str, int]:
        """Map dataset name -> integer OpenML ``tid`` (parsed from ``task_id_str``)."""
        if not self._tasks:
            return {}
        legacy = self.to_legacy_df()
        return legacy.drop_duplicates("dataset").set_index("dataset")["tid"].astype(int).to_dict()
