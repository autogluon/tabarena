from __future__ import annotations

import hashlib
import pickle
from collections import OrderedDict
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Annotated, Literal

import numpy as np
import openml
import openml._api_calls
import openml.utils
import pandas as pd
from openml.datasets.dataset import OpenMLDataset
from openml.datasets.functions import (
    _expand_parameter,
    _validated_data_attributes,
    attributes_arff_from_df,
)
from openml.tasks import (
    OpenMLClassificationTask,
    OpenMLRegressionTask,
    OpenMLSupervisedTask,
    TaskType,
)

SplitIndex = Annotated[str, "format: r{int}f{int}"]


@dataclass
class TabArenaTaskMetadata:
    """Metadata about the task to run.

    This metadata has different use cases for 1 or N elements in the splits_metadata.
    """

    dataset_name: str
    """Simple name of the dataset used for the task."""

    problem_type: str
    """The problem type of the task (e.g., 'binary', 'multiclass', 'regression')."""
    is_classification: bool
    """Whether the task is a classification task."""

    target_name: str
    """The name of the target variable in the dataset."""
    eval_metric: str
    """The evaluation metric used for the task."""

    """The number of splits in the task."""
    splits_metadata: dict[SplitIndex, SplitMetadata]
    """Mapping of split index to Metadata about the splits in the task."""

    stratify_on: str | None
    """The name of the column used for stratification during splitting."""
    group_on: str | None
    """The name of the column used for grouping during splitting."""
    time_on: str | None
    """The name of the column used for temporal splitting."""
    group_time_on: str | None
    """The name of the column that contains the time information for"""

    multiclass_min_n_classes_over_splits: int | None
    """The minimum number of classes across splits for classification tasks,
    None for regression tasks."""
    multiclass_max_n_classes_over_splits: int | None
    """The maximum number of classes across splits for classification tasks,
    None for regression tasks."""
    class_consistency_over_splits: bool | None
    """Whether the number of classes is consistent across splits for classification tasks,
    None for regression tasks."""

    tabarena_task_name: str | None
    """The name of the task used for TabArena. This is used for better identification
    and can be set by the user."""
    task_id_str: str | None
    """The task ID string used for the task. This functions as the unique identifier
     of the source of the metadat. This can either be an OpenML task ID, or it is the
     identifier of a local task (see `UserTask.task_id_str`).
    """

    @property
    def n_splits(self):
        """Get the number of splits in the task."""
        return len(self.splits_metadata)

    @property
    def split_indices(self) -> list[SplitIndex]:
        """Get a list of all split indices in the task."""
        return list(self.splits_metadata.keys())

    @property
    def split_index(self) -> SplitIndex:
        """Get the split index for the task.
        This is only supported for tasks with one split.
        """
        if self.n_splits != 1:
            raise ValueError(
                f"Cannot get split index for task with {self.n_splits} splits. "
                "This is only supported for tasks with exactly one split."
            )
        return self.split_indices[0]

    def to_dict(self, *, exclude_splits_metadata: bool = False) -> dict:
        """Convert the task metadata to a dictionary for better visualization."""
        res = asdict(self)
        if exclude_splits_metadata:
            res.pop("splits_metadata")
        return res

    def to_dataframe(self) -> pd.DataFrame:
        """Transform metadata to a DataFrame."""
        rows = []
        static_metadata = self.to_dict(exclude_splits_metadata=True)
        for split_metadata in self.splits_metadata.values():
            rows.append(
                {
                    **static_metadata,
                    **split_metadata.to_dict(),
                }
            )
        return pd.DataFrame(rows)

    @staticmethod
    def from_row(row: pd.Series) -> TabArenaTaskMetadata:
        """Reconstruct TabArenaTaskMetadata from a single dataframe row."""
        row_dict = row.to_dict()

        # Identify TabArenaTaskMetadata fields (excluding splits_metadata)
        task_field_names = {
            f.name for f in fields(TabArenaTaskMetadata) if f.name != "splits_metadata"
        }
        if not all(name in row_dict for name in task_field_names):
            raise ValueError(
                "Metadata row is missing required TabArenaTaskMetadata fields: "
                f"{task_field_names - row_dict.keys()}"
            )
        task_kwargs = {
            key: row_dict[key] for key in task_field_names if key in row_dict
        }

        # Identify SplitMetadata fields
        split_field_names = {f.name for f in fields(SplitMetadata)}
        if not all(name in row_dict for name in split_field_names):
            raise ValueError(
                "Metadata row is missing required SplitMetadata fields: "
                f"{split_field_names - row_dict.keys()}"
            )
        split_kwargs = {
            key: row_dict[key] for key in split_field_names if key in row_dict
        }
        # Reconstruct SplitMetadata
        split_metadata = SplitMetadata(**split_kwargs)

        # --- Construct final object ---
        return TabArenaTaskMetadata(
            **task_kwargs,
            splits_metadata={split_metadata.split_index: split_metadata},
        )

    def unroll_splits(self) -> list[TabArenaTaskMetadata]:
        """Unroll the TabArenaTaskMetadata into a list of TabArenaTaskMetadata instances
        each containing exactly one split in `splits_metadata`.
        """
        return [
            replace(
                self,
                splits_metadata={split_idx: split_meta},
            )
            for split_idx, split_meta in self.splits_metadata.items()
        ]


@dataclass
class SplitMetadata:
    """Metadata about the splits in the task."""

    repeat: int
    """The repeat index of the split.

    All splits with the same repeat index have the same data points,
    but split into folds differently.
    """
    fold: int
    """The fold index of the split."""
    num_instances_train: int
    """The number of training instances in the split."""
    num_instances_test: int
    """The number of test instances in the split."""
    num_classes: int
    """-1 for regression tasks, and maximal number of unique classes in the training
    and test set for classification tasks."""
    num_features_train: int
    """The number of features (excluding the target) in the training set of the split."""
    num_features_test: int
    """The number of features (excluding the target) in the test set of the split."""

    @staticmethod
    def get_split_index(*, repeat_i: int, fold_i: int) -> SplitIndex:
        """Get the split index for the given repeat and fold."""
        return f"r{repeat_i}f{fold_i}"

    @property
    def split_index(self) -> SplitIndex:
        """Get the split index for this split metadata."""
        return self.get_split_index(repeat_i=self.repeat, fold_i=self.fold)

    def to_dict(self) -> dict[str, int | str]:
        """Convert the split metadata to a dictionary."""
        res = asdict(self)
        res["split_index"] = self.split_index
        return res


class TabArenaTaskMetadataMixin:
    """A mixin class to add TabArena-specific metadata to OpenML tasks."""

    _task_metadata: TabArenaTaskMetadata | None = None

    def __init__(
        self,
        *,
        stratify_on: str | None = None,
        group_on: str | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
        **kwargs,
    ) -> None:
        """Checkout Data Foundry's PredictiveMLTaskMetadata for more information."""
        super().__init__(**kwargs)
        self.stratify_on = stratify_on
        self.group_on = group_on
        self.time_on = time_on
        self.group_time_on = group_time_on
        self._task_metadata = None

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

        splits_metadata = {}
        for repeat_i, splits in self.split.split.items():
            for fold_i, samples_for_split in splits.items():
                assert len(samples_for_split) == 1, (
                    "Only one sample per split is supported so far!."
                )
                train_idx, test_idx = samples_for_split[0]

                num_classes = -1
                if is_classification:
                    num_classes = max(
                        int(oml_dataset.iloc[train_idx][target_name].nunique()),
                        int(oml_dataset.iloc[test_idx][target_name].nunique()),
                    )

                # -1 for oml_task.target_name
                num_features_train = oml_dataset.iloc[train_idx].shape[1] - 1
                num_features_test = oml_dataset.iloc[test_idx].shape[1] - 1

                # Resolve problem type
                if num_classes == -1:
                    split_problem_type = "regression"
                elif num_classes == 2:
                    split_problem_type = "binary"
                    num_classes_list.append(num_classes)
                else:
                    split_problem_type = "multiclass"
                    num_classes_list.append(num_classes)
                if task_problem_type is None:
                    task_problem_type = split_problem_type
                else:
                    assert task_problem_type == split_problem_type, (
                        "All splits must have the same problem type."
                    )

                s_index = SplitMetadata.get_split_index(
                    repeat_i=repeat_i, fold_i=fold_i
                )
                splits_metadata[s_index] = SplitMetadata(
                    repeat=repeat_i,
                    fold=fold_i,
                    num_instances_train=len(train_idx),
                    num_instances_test=len(test_idx),
                    num_classes=num_classes,
                    num_features_train=num_features_train,
                    num_features_test=num_features_test,
                )

        if len(num_classes_list) == 0:
            min_n_classes = None
            max_n_classes = None
            class_consistency_over_splits = None
        else:
            min_n_classes = min(num_classes_list)
            max_n_classes = max(num_classes_list)
            class_consistency_over_splits = min_n_classes == max_n_classes

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
            tabarena_task_name=tabarena_task_name,
            task_id_str=task_id_str,
        )

        return self._task_metadata


class TabArenaOpenMLSupervisedTask(TabArenaTaskMetadataMixin, OpenMLSupervisedTask):
    """A local OpenMLSupervisedTask with additional metadata for TabArena."""


class TabArenaOpenMLClassificationTask(
    TabArenaTaskMetadataMixin, OpenMLClassificationTask
):
    """A local OpenMLClassificationTask with additional metadata for TabArena."""


class TabArenaOpenMLRegressionTask(TabArenaTaskMetadataMixin, OpenMLRegressionTask):
    """A local OpenMLRegressionTask with additional metadata for TabArena."""


# Patch Functions for OpenML Dataset
def _get_dataset(self, **kwargs) -> openml.datasets.OpenMLDataset:
    return self.local_dataset


class UserTask:
    """A user-defined task to run on custom datasets or tasks."""

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
        self._task_name_hash = hashlib.sha256(
            self.task_name.encode("utf-8")
        ).hexdigest()
        self._task_cache_path = task_cache_path

    @property
    def task_cache_path(self) -> Path:
        """Path to use for caching the local OpenML tasks."""
        if self._task_cache_path is not None:
            return self._task_cache_path
        return (
            (openml.config._resolve_default_cache_dir() / "tabarena_tasks")
            .expanduser()
            .resolve()
        )

    @staticmethod
    def from_task_id_str(task_id_str: str) -> UserTask:
        """Create a UserTask from a task ID string."""
        parts = task_id_str.split("|")
        if len(parts) != 4 or parts[0] != "UserTask":
            raise ValueError(f"Invalid task ID string: {task_id_str}")
        task_name = parts[2]
        task_cache_path = Path(parts[3])
        return UserTask(task_name=task_name, task_cache_path=task_cache_path)

    @property
    def task_id_str(self) -> str:
        """Task ID used for the task."""
        return f"UserTask|{self.task_id}|{self.task_name}|{self.task_cache_path}"

    @property
    def tabarena_task_name(self) -> str:
        """Task/Dataset Name used for the task/dataset."""
        return f"Task-{self.task_id}"

    @property
    def task_id(self) -> int:
        """Generate a unique task ID based on the task name and a UUID.
        This is used to identify the task, for example, when caching results.
        """
        return int(self._task_name_hash, 16) % 10**10

    @property
    def _local_dataset_id(self) -> str:
        return self._task_name_hash

    @property
    def _local_cache_path(self) -> Path:
        return (
            Path(openml.config._root_cache_directory)
            / "local"
            / "datasets"
            / self._local_dataset_id
        )

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
        group_on: str | None = None,
        time_on: str | None = None,
        group_time_on: str | None = None,
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
            The name of the column used for grouping during splitting.
        time_on:
            The name of the column used for temporal splitting.
        group_time_on:
            The name of the column that contains the time information for
            each group in case of grouped data.
        dataset_name:
            Name of the dataset. Must match OpenML allowed names.
            If None, a default name based on the task name is used.
        """
        dataset = deepcopy(dataset).reset_index(drop=True)
        self._validate_splits(splits=splits, n_samples=len(dataset))

        task_type = (
            TaskType.SUPERVISED_CLASSIFICATION
            if problem_type == "classification"
            else TaskType.SUPERVISED_REGRESSION
        )
        extra_kwargs = {}
        if task_type == TaskType.SUPERVISED_CLASSIFICATION:
            task_cls = TabArenaOpenMLClassificationTask
            extra_kwargs["class_labels"] = list(np.unique(dataset[target_feature]))
        elif task_type == TaskType.SUPERVISED_REGRESSION:
            task_cls = TabArenaOpenMLRegressionTask
        else:
            raise NotImplementedError(f"Task type {task_type:d} not supported.")

        if dataset_name is None:
            dataset_name = f"Dataset-{self.task_name}"
        print(
            f"Creating local OpenML task {self.task_id} with dataset '{dataset_name}'..."
        )
        local_dataset = openml_create_datasets_without_arff_dump(
            name=dataset_name,
            data=dataset,
            default_target_attribute=target_feature,
        )
        # Cache data to disk
        parquet_file = self._local_cache_path / "data.pq"
        parquet_file.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(parquet_file)
        del dataset  # Free memory

        # We only need local_dataset.get_data() from the OpenMLDataset, thus, we make
        # sure with the code below that get_data() returns the data.
        local_dataset.parquet_file = parquet_file
        local_dataset.data_file = "ignored"  # not used for local datasets

        # Create the task
        task = task_cls(
            stratify_on=stratify_on,
            group_on=group_on,
            time_on=time_on,
            group_time_on=group_time_on,
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
    def _validate_splits(
        *, splits: dict[int, dict[int, tuple[list, list]]], n_samples: int
    ) -> None:
        """Validate the splits passed by the user."""
        if not isinstance(splits, dict):
            raise ValueError("Splits must be a dictionary.")

        found_structure = None
        for repeat_id, split_dict in splits.items():
            if not isinstance(split_dict, dict):
                raise ValueError(f"Splits for repeat {repeat_id} must be a dictionary.")
            test_indices_per_repeat = set()
            for split_id, (train_indices, test_indices) in split_dict.items():
                if not isinstance(train_indices, list) or not isinstance(
                    test_indices, list
                ):
                    raise ValueError(f"Indices for split {split_id} must be lists.")
                if not all(
                    isinstance(idx, int) for idx in train_indices + test_indices
                ):
                    raise ValueError(
                        f"All indices in split {split_id} must be integers."
                    )
                if len(train_indices) == 0 or len(test_indices) == 0:
                    raise ValueError(
                        f"Train and test indices in split {split_id} must not be empty."
                    )
                if set(train_indices) & set(test_indices):
                    raise ValueError(
                        f"Train and test indices in split {split_id} must not overlap."
                    )
                if any(np.array(train_indices + test_indices) < 0):
                    raise ValueError(
                        f"Indices in split {split_id} must be non-negative."
                    )
                if any(np.array(train_indices + test_indices) >= n_samples):
                    raise ValueError(
                        f"Indices in split {split_id} must not exceed the dataset size (0 to {n_samples - 1})."
                    )
                if test_indices_per_repeat & set(test_indices):
                    raise ValueError(
                        f"Test indices in split {split_id} must not overlap with previous splits in repeat {repeat_id}."
                    )
                test_indices_per_repeat.update(test_indices)

            if found_structure is None:
                found_structure = len(split_dict)
            elif found_structure != len(split_dict):
                raise ValueError("All repeats must have the same number of splits.")

    @property
    def openml_task_path(self) -> Path:
        return self.task_cache_path / f"{self.task_id}.pkl"

    def save_local_openml_task(self, task: OpenMLSupervisedTask) -> None:
        """Safe the OpenML task to be usable by loading from disk later."""
        print(f"Saving local task {self.task_name} to: {self.task_cache_path}")

        self.task_cache_path.mkdir(parents=True, exist_ok=True)
        # Remove monkey patch to avoid pickle issues.
        del task.get_dataset
        with self.openml_task_path.open("wb") as f:
            pickle.dump(task, f)

    def load_local_openml_task(self) -> TabArenaOpenMLSupervisedTask:
        """Load a local OpenML task from disk."""
        if not self.openml_task_path.exists():
            raise FileNotFoundError(
                f"Cached task file {self.openml_task_path} does not exist!"
            )

        with self.openml_task_path.open("rb") as f:
            task: OpenMLSupervisedTask = pickle.load(f)
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

    # infer the type of data for each column of the DataFrame
    attributes_ = attributes_arff_from_df(data)

    ignore_attributes = _expand_parameter(ignore_attribute)
    _validated_data_attributes(ignore_attributes, attributes_, "ignore_attribute")

    default_target_attributes = _expand_parameter(default_target_attribute)
    _validated_data_attributes(
        default_target_attributes, attributes_, "default_target_attribute"
    )

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
    sklearn_splits: Iterable, n_splits: int
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
