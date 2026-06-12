from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import numpy as np
import openml
import openml._api_calls
import openml.utils
import pandas as pd
from loguru import logger
from openml.datasets.dataset import OpenMLDataset
from openml.datasets.functions import (
    _expand_parameter,
    _validated_data_attributes,
    attributes_arff_from_df,
)
from openml.tasks import (
    OpenMLSupervisedTask,
    TaskType,
)

from tabarena.benchmark.task.openml import (
    TabArenaOpenMLClassificationTask,
    TabArenaOpenMLRegressionTask,
    TabArenaOpenMLSupervisedTask,
)
from tabarena.benchmark.task.spec import TaskSpec, register_task_spec_parser

if TYPE_CHECKING:
    from collections.abc import Iterable

    from tabarena.benchmark.task.in_memory import InMemoryTaskWrapper
    from tabarena.benchmark.task.metadata import (
        GroupLabelTypes,
        SplitTimeHorizonTypes,
        SplitTimeHorizonUnitTypes,
    )
    from tabarena.benchmark.task.wrapper import TaskWrapper


#: Number of hex chars of the task-name hash appended to a slug for uniqueness.
_SLUG_HASH_LEN = 12

#: Marker of the native task persistence format (see :meth:`UserTask.save_task`).
_USER_TASK_FORMAT = "tabarena-user-task-v1"


def _slugify(name: str, *, max_len: int = 40) -> str:
    """Make ``name`` filesystem/display safe: chars outside ``[A-Za-z0-9._-]`` -> ``_``.

    Trimmed of leading/trailing ``_.-`` and length-capped; falls back to ``"task"``
    if nothing readable remains.
    """
    safe = "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)
    return safe.strip("_.-")[:max_len] or "task"


# Patch Functions for OpenML Dataset
def _get_dataset(self, **kwargs) -> openml.datasets.OpenMLDataset:
    return self.local_dataset


class UserTask(TaskSpec):
    """A user-defined task to run on custom datasets or tasks.

    As a :class:`TaskSpec`, a ``UserTask`` vends itself from local disk: ``load``
    reads the task cached by :meth:`save_local_openml_task` — no remote service
    involved.
    """

    def __init__(self, *, task_name: str, task_cache_path: Path | None = None) -> None:
        """Initialize a user-defined task.

        NOTE: do not store any attributes in this class but put them
        in the local task created from this class, as this class
        is only used to create/load the task.

        Parameters
        ----------
        task_name: str
            The name of the task. Make sure this is a unique name on your system,
            as we create the cache based on this name.
        task_cache_path: Path | None, default=None
            Path to use for caching the local OpenML tasks.
            If None, the default OpenML cache directory is used.
        """
        self.task_name = task_name
        self._task_name_hash = hashlib.sha256(self.task_name.encode("utf-8")).hexdigest()
        self._task_cache_path = task_cache_path

    @property
    def task_cache_path(self) -> Path:
        """Path to use for caching the local OpenML tasks."""
        if self._task_cache_path is not None:
            return self._task_cache_path
        return (openml.config._root_cache_directory / "tabarena_tasks").expanduser().resolve()

    @staticmethod
    def from_task_id_str(task_id_str: str) -> UserTask:
        """Reconstruct a UserTask from its ``task_id_str``.

        Two forms are accepted:

        * ``UserTask|{task_id}|{task_name}`` — the standardized, portable form.
          The cache directory is resolved from the ambient OpenML config at load
          time (``<openml_root>/tabarena_tasks``), so the same id works on any
          machine. This is what data_foundry / BeyondArena tasks use.
        * ``UserTask|{task_id}|{task_name}|{task_cache_path}`` — the explicit-path
          fallback, for a task stored at a custom location. The embedded path is
          honored as-is.

        ``{task_id}`` is informational only; it is recomputed from ``{task_name}``.
        """
        parts = task_id_str.split("|")
        if (parts[0] != "UserTask") or (len(parts) not in (3, 4)):
            raise ValueError(f"Invalid task ID string: {task_id_str}")
        task_name = parts[2]
        task_cache_path = Path(parts[3]) if len(parts) == 4 else None
        return UserTask(task_name=task_name, task_cache_path=task_cache_path)

    @property
    def task_id_str(self) -> str:
        """Portable identifier for this task.

        Standardized tasks (no explicit ``task_cache_path``) serialize *without* a
        path, so the id is machine-independent and resolves against the ambient
        OpenML cache on load. Tasks created with a custom ``task_cache_path``
        include it so they remain self-describing.
        """
        if self._task_cache_path is None:
            return f"UserTask|{self.task_id}|{self.task_name}"
        return f"UserTask|{self.task_id}|{self.task_name}|{self.task_cache_path}"

    @property
    def slug(self) -> str:
        """Human-readable, filesystem-safe, unique identifier for the task.

        Format ``{readable}-{shorthash}`` where ``readable`` is the first
        ``/``-segment of the task name (the dataset name for data_foundry tasks,
        e.g. ``blood_transfusion/<uuid>`` -> ``blood_transfusion``) and
        ``shorthash`` is the first :data:`_SLUG_HASH_LEN` hex chars of the task
        name's SHA-256. The hash is always present and derived from the full
        (unique) task name, so the slug stays unique and collision-safe even when
        the readable parts coincide, and is reproducible from ``task_name`` alone.
        """
        readable = _slugify(self.task_name.split("/")[0])
        return f"{readable}-{self._task_name_hash[:_SLUG_HASH_LEN]}"

    @property
    def tabarena_task_name(self) -> str:
        """Task/Dataset name used for the task (and as the results ``dataset`` key)."""
        return self.slug

    @property
    def cache_key(self) -> str:
        """Filesystem-safe results/text-cache key for this task (the :attr:`slug`)."""
        return self.slug

    def load(self) -> TaskWrapper:
        """Vend the task from local disk (see :meth:`save_task`).

        A task persisted in the native format loads as an
        :class:`~tabarena.benchmark.task.in_memory.InMemoryTaskWrapper`; a legacy
        cache (a pickled local OpenML task, see :meth:`save_local_openml_task`)
        loads through the OpenML-backed wrapper. Either way the task's own eval
        metric (when set at creation) is honored, and data is lazy-loaded: the
        engine re-reads the local cache per data access instead of keeping the
        full frame resident. An attached collection entry (``task_metadata``)
        takes precedence over the persisted metadata.
        """
        payload = self._read_task_file()

        if isinstance(payload, dict) and payload.get("format") == _USER_TASK_FORMAT:
            from tabarena.benchmark.task.in_memory import InMemoryTaskWrapper

            task_path = self.task_path

            def _read_dataset() -> pd.DataFrame:
                with task_path.open("rb") as f:
                    return pickle.load(f)["dataset"]

            return InMemoryTaskWrapper(
                dataset=_read_dataset,
                splits=payload["splits"],
                metadata=self.task_metadata if self.task_metadata is not None else payload["metadata"],
                lazy_load_data=True,
            )

        # Legacy cache: a pickled local OpenML task object.
        from tabarena.benchmark.task.openml import OpenMLTaskWrapper

        payload.get_dataset = _get_dataset.__get__(payload, OpenMLSupervisedTask)
        return OpenMLTaskWrapper(
            task=payload,
            use_task_eval_metric=True,
            lazy_load_data=True,
            metadata=self.task_metadata,
        )

    def _read_task_file(self):
        """Unpickle this task's cache file (native payload dict or legacy OpenML task)."""
        if not self.task_path.exists():
            raise FileNotFoundError(f"Cached task file {self.task_path} does not exist!")
        with self.task_path.open("rb") as f:
            return pickle.load(f)

    def resolve_task_name(self, task: TaskWrapper) -> str:
        """The results ``dataset`` key — known up front (the loaded task is not needed)."""
        return self.tabarena_task_name

    @property
    def task_id(self) -> int:
        """Generate a unique integer task ID based on the task name.

        Used where an integer id is structurally required (the OpenML task object).
        Human-facing references / caches use :attr:`slug` instead.
        """
        return int(self._task_name_hash, 16) % 10**10

    @property
    def _local_dataset_id(self) -> str:
        return self._task_name_hash

    @property
    def _local_cache_path(self) -> Path:
        return Path(openml.config._root_cache_directory) / "local" / "datasets" / self._local_dataset_id

    def get_dataset_name(self, dataset_name: str | None = None) -> str:
        """Get the dataset name to use for the local OpenML dataset."""
        if dataset_name is not None:
            return dataset_name
        return f"Dataset-{self.task_name}"

    # TODO: support local OpenML tasks inside of OpenML code...
    def create_local_openml_task(
        self,
        *,
        target_feature: str,
        problem_type: Literal["classification", "regression"],
        dataset: pd.DataFrame,
        splits: dict[int, dict[int, tuple[list, list]]],
        eval_metric: str | None = None,
        stratify_on: str | None = None,
        group_on: str | list[str] | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        group_labels: GroupLabelTypes | None = None,
        split_time_horizon: SplitTimeHorizonTypes | None = None,
        split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
        dataset_name: str | None = None,
    ) -> OpenMLSupervisedTask:
        """Convert the user-defined task to a local (unpublished) OpenMLSupervisedTask.

        Parameters
        ----------
        dataset: pd.DataFrame
            The dataset to be used for the task. It should be a pandas DataFrame
            with features and target variable. Moreover, make sure the DataFrame
            has the correct dtypes for each column, as this will be used
            to infer the metadata of features. Thus, make sure that:
                - Numerical features are of a number type.
                - Categorical features are of type 'category'.
                - Text features are of a string type.
                - Date features are of a date type.
        target_feature: str
            The name of the target feature in the dataset. This must be a column
            in the dataset DataFrame.
        problem_type: Literal['classification', 'regression']
            The type of problem to be solved. It can be either 'classification'
            or 'regression'.
        splits: dict[int, dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]]
            A dictionary the train-tests splits per repeat and fold.
            These splits represent the outer splits that are used to evaluate models,
            and not the inner splits used for tuning/validation/HPO.

            The structure is:
            {
                repeat_id: {
                    split_id: {
                        (train_indices, test_indices)
                    }
                    ...
                }
                ...
            }
            where train_indices and test_indices are lists of indices, starting from 0.

            Note the following assumptions:
                - The indices in train_indices and test_indices do not overlap.
                - Per repeat_id, one can have multiple split_ids, but the test_indices
                  should not overlap across split_ids.
                - Splits across repeat_ids should have the same structure (e.g., if
                  there is only one split in repeat_id 0, there should be only one split
                  in all other repeat_ids).
        eval_metric: str | None, default=None
            If None, we pass None to the OpenML task and later the default
            TabArena metrics are used.
            Otherwise, the metric specified here is used for evaluating the task.
            Note that the metric must be registered in TabArena/AutoGluon.
        stratify_on:
            The name of the column used for stratification during splitting.
        group_on:
            The name(s) of the column used for grouping during splitting.
        time_on:
            The name of the column used for temporal splitting.
        group_time_on:
            The name of the column that contains the time information for
            each group in case of grouped data.
        split_time_horizon:
            The time horizon used for temporal splitting. This can be a number (e.g., 5).
        split_time_horizon_unit:
            The unit of the time horizon used for temporal splitting. This can be a string (e.g., 'days').
        dataset_name:
            Name of the dataset. Must match OpenML allowed names.
            If None, a default name based on the task name is used.
        """
        dataset = deepcopy(dataset).reset_index(drop=True)
        self._validate_splits(splits=splits, n_samples=len(dataset))

        task_type = (
            TaskType.SUPERVISED_CLASSIFICATION if problem_type == "classification" else TaskType.SUPERVISED_REGRESSION
        )
        extra_kwargs = {}
        if task_type == TaskType.SUPERVISED_CLASSIFICATION:
            task_cls = TabArenaOpenMLClassificationTask
            extra_kwargs["class_labels"] = list(np.unique(dataset[target_feature]))
        elif task_type == TaskType.SUPERVISED_REGRESSION:
            task_cls = TabArenaOpenMLRegressionTask
        else:
            raise NotImplementedError(f"Task type {task_type:d} not supported.")

        dataset_name = self.get_dataset_name(dataset_name=dataset_name)
        logger.debug(f"Creating local OpenML task {self.task_id} with dataset '{dataset_name}'...")
        local_dataset = openml_create_datasets_without_arff_dump(
            name=dataset_name,
            data=dataset,
            default_target_attribute=target_feature,
        )
        # Cache data to disk
        #   This ensures to keep the dtypes of the original dataframe (and not lose it via parquet or similar)
        #   Moreover, this skips that OpenML itself has do pickle dump the dataset again.
        pickle_file = self._local_cache_path / "data.pkl.py3"
        pickle_file.parent.mkdir(parents=True, exist_ok=True)
        with pickle_file.open("wb") as fh:
            pickle.dump(
                (dataset, [dataset[c].dtype.name == "category" for c in dataset.columns], list(dataset.columns)),
                fh,
                pickle.HIGHEST_PROTOCOL,
            )
        del dataset  # Free memory

        # We only need local_dataset.get_data() from the OpenMLDataset, thus, we make
        # sure with the code below that get_data() returns the data.
        local_dataset.data_pickle_file = pickle_file
        local_dataset.cache_format = "pickle"
        local_dataset.data_file = "ignored"  # not used for local datasets

        # Create the task
        task = task_cls(
            stratify_on=stratify_on,
            group_on=group_on,
            time_on=time_on,
            group_time_on=group_time_on,
            group_labels=group_labels,
            split_time_horizon=split_time_horizon,
            split_time_horizon_unit=split_time_horizon_unit,
            task_id=self.task_id,
            task_type_id=task_type,
            task_type="None",  # Placeholder, not used for local tasks
            data_set_id=-1,  # Placeholder, not used for local tasks
            target_name=target_feature,
            evaluation_measure=eval_metric,
            **extra_kwargs,
        )
        task.local_dataset = local_dataset
        task.get_dataset = _get_dataset.__get__(task, OpenMLSupervisedTask)

        # Transform TabArena splits to OpenMLSplit format
        openml_splits = {}
        for repeat in splits:
            openml_splits[repeat] = OrderedDict()
            for fold in splits[repeat]:
                openml_splits[repeat][fold] = OrderedDict()
                # We do not support learning curves tasks, so no need for samples.
                openml_splits[repeat][fold][0] = (
                    np.array(splits[repeat][fold][0], dtype=int),
                    np.array(splits[repeat][fold][1], dtype=int),
                )

        task.split = openml.tasks.split.OpenMLSplit(
            name="User-Splits",
            description="User-defined splits for a custom task.",
            split=openml_splits,
        )

        return task

    @staticmethod
    def _validate_splits(*, splits: dict[int, dict[int, tuple[list, list]]], n_samples: int) -> None:
        """Validate the splits passed by the user."""
        if not isinstance(splits, dict):
            raise ValueError("Splits must be a dictionary.")

        found_structure = None
        for repeat_id, split_dict in splits.items():
            if not isinstance(split_dict, dict):
                raise ValueError(f"Splits for repeat {repeat_id} must be a dictionary.")
            test_indices_per_repeat = set()
            for split_id, (train_indices, test_indices) in split_dict.items():
                if not isinstance(train_indices, list) or not isinstance(test_indices, list):
                    raise ValueError(f"Indices for split {split_id} must be lists.")
                if not all(isinstance(idx, int) for idx in train_indices + test_indices):
                    raise ValueError(f"All indices in split {split_id} must be integers.")
                if len(train_indices) == 0 or len(test_indices) == 0:
                    raise ValueError(f"Train and test indices in split {split_id} must not be empty.")
                if set(train_indices) & set(test_indices):
                    raise ValueError(f"Train and test indices in split {split_id} must not overlap.")
                if any(np.array(train_indices + test_indices) < 0):
                    raise ValueError(f"Indices in split {split_id} must be non-negative.")
                if any(np.array(train_indices + test_indices) >= n_samples):
                    raise ValueError(
                        f"Indices in split {split_id} must not exceed the dataset size (0 to {n_samples - 1}).",
                    )
                if test_indices_per_repeat & set(test_indices):
                    raise ValueError(
                        f"Test indices in split {split_id} must not overlap with previous splits in repeat {repeat_id}.",
                    )
                test_indices_per_repeat.update(test_indices)

            if found_structure is None:
                found_structure = len(split_dict)
            elif found_structure != len(split_dict):
                raise ValueError("All repeats must have the same number of splits.")

    # --- Native task creation / persistence ------------------------------------------
    def create_task(
        self,
        *,
        target_feature: str,
        problem_type: Literal["classification", "regression"],
        dataset: pd.DataFrame,
        splits: dict[int, dict[int, tuple[list, list]]],
        eval_metric: str | None = None,
        stratify_on: str | None = None,
        group_on: str | list[str] | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        group_labels: GroupLabelTypes | None = None,
        split_time_horizon: SplitTimeHorizonTypes | None = None,
        split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None,
        dataset_name: str | None = None,
    ) -> InMemoryTaskWrapper:
        """Build the task natively as an ``InMemoryTaskWrapper`` — no OpenML objects.

        Takes the same inputs as :meth:`create_local_openml_task` (see there for the
        parameter documentation) but returns a runnable
        :class:`~tabarena.benchmark.task.in_memory.InMemoryTaskWrapper` whose
        ``metadata`` is the task's exact ``TabArenaTaskMetadata`` — computed here,
        with this ``UserTask``'s identity (``tabarena_task_name`` / ``task_id_str``)
        filled in. Persist with :meth:`save_task`; reload with :meth:`load`.
        """
        if problem_type not in ("classification", "regression"):
            raise NotImplementedError(f"Problem type {problem_type!r} not supported.")
        dataset = deepcopy(dataset).reset_index(drop=True)
        self._validate_splits(splits=splits, n_samples=len(dataset))
        splits_arrays = {
            repeat: {
                fold: (np.asarray(train_idx, dtype=int), np.asarray(test_idx, dtype=int))
                for fold, (train_idx, test_idx) in fold_splits.items()
            }
            for repeat, fold_splits in splits.items()
        }

        from tabarena.benchmark.task.in_memory import InMemoryTaskWrapper
        from tabarena.benchmark.task.metadata.compute import compute_task_metadata

        metadata = compute_task_metadata(
            dataset=dataset,
            dataset_name=self.get_dataset_name(dataset_name=dataset_name),
            target_name=target_feature,
            is_classification=problem_type == "classification",
            eval_metric=eval_metric,
            splits=splits_arrays,
            stratify_on=stratify_on,
            group_on=group_on,
            time_on=time_on,
            group_time_on=group_time_on,
            group_labels=group_labels,
            split_time_horizon=split_time_horizon,
            split_time_horizon_unit=split_time_horizon_unit,
            tabarena_task_name=self.tabarena_task_name,
            task_id_str=self.task_id_str,
        )
        return InMemoryTaskWrapper(dataset=dataset, splits=splits_arrays, metadata=metadata)

    def save_task(self, task: InMemoryTaskWrapper) -> None:
        """Persist a task built by :meth:`create_task` to this task's cache file.

        The native format is one pickle holding ``(dataset, splits, metadata)`` —
        dtypes survive exactly, and no OpenML object is involved. :meth:`load`
        re-vends it as an ``InMemoryTaskWrapper`` (lazy-loading from this file).
        """
        logger.debug(f"Saving task {self.task_name} to: {self.task_cache_path}")
        dataset = task._dataset_source() if callable(task._dataset_source) else task._dataset_source
        payload = {
            "format": _USER_TASK_FORMAT,
            "dataset": dataset,
            "splits": task._splits,
            "metadata": task.metadata,
        }
        self.task_cache_path.mkdir(parents=True, exist_ok=True)
        with self.task_path.open("wb") as f:
            pickle.dump(payload, f, pickle.HIGHEST_PROTOCOL)

    @property
    def task_path(self) -> Path:
        """This task's cache file (native payload, or a legacy pickled OpenML task)."""
        return self.task_cache_path / f"{self.slug}.pkl"

    # --- Legacy local-OpenML-task persistence -----------------------------------------
    @property
    def openml_task_path(self) -> Path:
        """Deprecated alias of :attr:`task_path` (kept for existing callers)."""
        return self.task_path

    def save_local_openml_task(self, task: OpenMLSupervisedTask) -> None:
        """Safe the OpenML task to be usable by loading from disk later."""
        logger.debug(f"Saving local task {self.task_name} to: {self.task_cache_path}")

        self.task_cache_path.mkdir(parents=True, exist_ok=True)
        # Remove monkey patch to avoid pickle issues.
        del task.get_dataset
        with self.task_path.open("wb") as f:
            pickle.dump(task, f)

    def load_local_openml_task(self) -> TabArenaOpenMLSupervisedTask:
        """Load a local OpenML task from disk (legacy caches only; see :meth:`load`)."""
        task = self._read_task_file()
        if isinstance(task, dict):
            raise TypeError(
                f"Task {self.task_name!r} is persisted in the native TabArena format "
                f"({task.get('format')!r}); there is no OpenML task object to load — use `UserTask.load()`.",
            )
        # Add monkey patch again.
        task.get_dataset = _get_dataset.__get__(task, OpenMLSupervisedTask)

        return task


def openml_create_datasets_without_arff_dump(
    *,
    name: str,
    data: pd.DataFrame,
    default_target_attribute: str,
) -> OpenMLDataset:
    """Custom version of from openml.datasets.functions import create_dataset
    to improve local task creation and avoid ARFF slowdown.
    """
    assert isinstance(data, pd.DataFrame)
    description = None
    creator = None
    contributor = None
    collection_date = None
    language = None
    licence = None
    ignore_attribute = None
    citation = "N/A"
    row_id_attribute = None
    original_data_url = None
    paper_url = None
    version_label = None
    update_comment = None

    # infer the row id from the index of the dataset
    if row_id_attribute is None:
        row_id_attribute = data.index.name
    # When calling data.values, the index will be skipped.
    # We need to reset the index such that it is part of the data.
    if data.index.name is not None:
        data = data.reset_index()

    # liac-arff only supports integer, floating, string, categorical, and boolean dtypes.
    # Types like datetime64, timedelta64, period, and interval are unsupported and will
    # raise a ValueError in attributes_arff_from_df.
    # We cast such columns to string here solely so that attributes_arff_from_df can
    # infer ARFF attribute metadata. This does NOT affect the actual stored data — the
    # dataset is persisted as parquet (see caller) and loaded from there, never from ARFF.
    unsupported_cols = data.select_dtypes(include=["datetime64", "timedelta64"]).columns
    # select_dtypes doesn't support "period" or "interval" as strings, so detect manually
    unsupported_cols = unsupported_cols.append(
        pd.Index(col for col in data.columns if isinstance(data[col].dtype, (pd.PeriodDtype, pd.IntervalDtype))),
    )
    # Cast categories of categorical columns to string so that
    # attributes_arff_from_df can handle them (e.g. integer categories).
    cat_cols_to_fix = [
        col
        for col in data.select_dtypes(include=["category"]).columns
        if not pd.api.types.is_string_dtype(data[col].cat.categories)
    ]

    if len(unsupported_cols) > 0 or len(cat_cols_to_fix) > 0:
        data = data.copy()
    if len(unsupported_cols) > 0:
        data[unsupported_cols] = data[unsupported_cols].astype(str)
    for col in cat_cols_to_fix:
        data[col] = data[col].cat.rename_categories(str)

    # infer the type of data for each column of the DataFrame
    attributes_ = attributes_arff_from_df(data)

    ignore_attributes = _expand_parameter(ignore_attribute)
    _validated_data_attributes(ignore_attributes, attributes_, "ignore_attribute")

    default_target_attributes = _expand_parameter(default_target_attribute)
    _validated_data_attributes(default_target_attributes, attributes_, "default_target_attribute")

    return OpenMLDataset(
        name=name,
        description=description,
        creator=creator,
        contributor=contributor,
        collection_date=collection_date,
        language=language,
        licence=licence,
        default_target_attribute=default_target_attribute,
        row_id_attribute=row_id_attribute,
        ignore_attribute=ignore_attribute,
        citation=citation,
        version_label=version_label,
        original_data_url=original_data_url,
        paper_url=paper_url,
        update_comment=update_comment,
        # Skip unused ARFF usage for local datasets
        data_format="arff",
        dataset=None,
    )


def from_sklearn_splits_to_user_task_splits(
    sklearn_splits: Iterable,
    n_splits: int,
) -> dict[int, dict[int, tuple[list, list]]]:
    """Convert sklearn splits to the OpenML splits format used in TabArena's
    local user tasks.

    Arguments:
    ---------
    sklearn_splits: Iterable
        An iterable of (train_indices, test_indices) tuples as returned by
        sklearn's splitters (e.g., RepeatedKFold, ...).
    n_splits: int
        The number of splits per repeat (e.g., for RepeatedKFold).

    Returns:
    -------
    splits: dict[int, dict[int, tuple[list, list]]]
        A dictionary the train-tests splits per repeat and fold in
        the format of OpenML.
    """
    splits = {}
    for split_i, (train_idx, test_idx) in enumerate(sklearn_splits):
        repeat_i = split_i // n_splits
        fold_i = split_i % n_splits
        if repeat_i not in splits:
            splits[repeat_i] = {}
        splits[repeat_i][fold_i] = (train_idx.tolist(), test_idx.tolist())
    return splits


# Serialized UserTask ids ("UserTask|...") reconstruct through the spec registry.
register_task_spec_parser("UserTask", UserTask.from_task_id_str)
