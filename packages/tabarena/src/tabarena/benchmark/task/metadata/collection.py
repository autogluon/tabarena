from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING

import pandas as pd

from tabarena.benchmark.task.metadata.schema import (
    SplitMetadata,
    TabArenaTaskMetadata,
    derive_task_type,
    tid_from_task_id_str,
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

    def subset(self, dataset_fold_repeats: list[tuple[str, int, int]]) -> TaskMetadataCollection:
        """A new collection restricted to the given ``(dataset, fold, repeat)`` triplets.

        Each task keeps only the splits whose ``(fold, repeat)`` was requested for its
        dataset; tasks left with no splits are dropped. Every requested triplet must be a
        real split of this collection (raises otherwise), so the result's
        :meth:`dataset_fold_repeats` is exactly the requested set. Useful for scoping a
        run to a subset of splits without a separate "allowed triplets" channel — the
        collection itself becomes the single source of truth for what to run.
        """
        valid = set(self.dataset_fold_repeats())
        invalid = [t for t in dataset_fold_repeats if t not in valid]
        if invalid:
            invalid_str = "\n\t".join(str(x) for x in invalid)
            raise ValueError(
                f"{len(invalid)} requested (dataset, fold, repeat) triplet(s) are not splits of this "
                f"collection:\n\t{invalid_str}",
            )

        wanted: dict[str, set[tuple[int, int]]] = {}
        for dataset, fold, repeat in dataset_fold_repeats:
            wanted.setdefault(dataset, set()).add((fold, repeat))

        new_tasks: list[TabArenaTaskMetadata] = []
        for t in self._tasks:
            wanted_splits = wanted.get(t.tabarena_task_name)
            if not wanted_splits:
                continue
            kept = {idx: s for idx, s in t.splits_metadata.items() if (s.fold, s.repeat) in wanted_splits}
            if kept:
                new_tasks.append(replace(t, splits_metadata=kept))
        return TaskMetadataCollection(new_tasks)

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

    def task_grid(self) -> pd.DataFrame:
        """One row per ``(dataset, fold, repeat, split)`` carrying the subset-predicate metadata.

        This is the frame the subset predicates evaluate against (see
        :attr:`~tabarena.nips2025_utils.tabarena_context.TabArenaContext.SUBSET_PREDICATES`): a
        task-level view enumerated natively from each task's ``splits_metadata``, joined with the
        dataset-level predicate columns.

        Columns:

        * ``dataset`` / ``fold`` / ``repeat`` — task identity (raw fold/repeat).
        * ``split`` — ``n_folds * repeat + fold`` (``n_folds`` = max fold + 1 per dataset); this is
          what ``"lite"`` keys on (``split == 0``) and what a results frame's ``fold`` column maps
          to when subsetting.
        * predicate columns, using the predicate-facing names: ``max_train_rows`` (mean per-fold
          train size over the dataset's splits — matches :meth:`to_legacy_df`'s
          ``n_samples_train_per_fold``), ``n_features`` (``num_features``), ``n_classes``
          (``num_classes``), ``problem_type``.
        """
        cols = ["dataset", "fold", "repeat", "split", "max_train_rows", "n_features", "n_classes", "problem_type"]
        n_folds_by_dataset: dict[str, int] = {}
        train_sizes: dict[str, list[float]] = {}
        meta: dict[str, dict] = {}
        for t in self._tasks:
            ds = t.tabarena_task_name
            meta.setdefault(
                ds,
                {"n_features": t.num_features, "n_classes": t.num_classes, "problem_type": t.problem_type},
            )
            for split in t.splits_metadata.values():
                n_folds_by_dataset[ds] = max(n_folds_by_dataset.get(ds, 0), split.fold + 1)
                train_sizes.setdefault(ds, []).append(split.num_instances_train)
        mean_train = {ds: (sum(sizes) / len(sizes)) for ds, sizes in train_sizes.items()}
        rows = [
            {
                "dataset": ds,
                "fold": fold,
                "repeat": repeat,
                "split": n_folds_by_dataset[ds] * repeat + fold,
                "max_train_rows": mean_train[ds],
                "n_features": meta[ds]["n_features"],
                "n_classes": meta[ds]["n_classes"],
                "problem_type": meta[ds]["problem_type"],
            }
            for ds, fold, repeat in self.dataset_fold_repeats()
        ]
        return pd.DataFrame(rows, columns=cols)

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
          ``splits_metadata`` grid (per-split sizes from ``n_samples_train/test_per_fold``,
          kept as floats so they round-trip through :meth:`to_legacy_df` without truncation).
        * normalized: ``num_classes`` is set to the schema's ``-1`` for regression (rather than
          the legacy ``0``), so the rebuilt collection matches the native convention.
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
            # Normalize to the schema's regression convention (-1) rather than preserving the
            # legacy 0, so the rebuilt collection matches the native bundle.
            n_classes = -1 if problem_type == "regression" else int(row["n_classes"])
            num_features = int(row["n_features"])
            num_instances = int(row["NumberOfInstances"])
            # Kept as floats: the legacy frame stores a (possibly fractional) per-fold average,
            # so int() would truncate it. to_legacy_df recovers it via a float mean, so the
            # value round-trips exactly.
            n_train = float(row["n_samples_train_per_fold"])
            n_test = float(row["n_samples_test_per_fold"])
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
        """Map dataset name -> integer OpenML ``tid``, parsed natively from each task's
        ``task_id_str`` (first task per dataset wins; no legacy-frame round-trip).
        """
        out: dict[str, int] = {}
        for t in self._tasks:
            out.setdefault(t.tabarena_task_name, tid_from_task_id_str(t.task_id_str))
        return out
