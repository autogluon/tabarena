from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields, replace
from enum import StrEnum
from typing import Annotated, Literal

import pandas as pd

SplitIndex = Annotated[str, "format: r{int}f{int}"]

SplitTimeHorizonTypes = str | int | float
SplitTimeHorizonUnitTypes = Literal["steps", "days", "weeks", "months", "years"] | str


class GroupLabelTypes(StrEnum):
    PER_SAMPLE = "per_sample"
    PER_GROUP = "per_group"


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
    split_time_horizon: SplitTimeHorizonTypes | None
    """The time horizon used for temporal splitting. This can be a number (e.g., 5)."""
    split_time_horizon_unit: SplitTimeHorizonUnitTypes | None
    """The unit of the time horizon used for temporal splitting. This can be a string (e.g., 'days') or one of
    the predefined literals."""

    stratify_on: str | None
    """The name of the column used for stratification during splitting."""
    time_on: str | None
    """The name of the column used for temporal splitting."""
    group_on: str | list[str] | None
    """The name of the column used for grouping during splitting."""
    group_time_on: str | None
    """The name of the column that contains the time information for"""
    group_labels: GroupLabelTypes | None
    """Whether the group_on column(s) contain labels for each sample, or for each group."""

    multiclass_min_n_classes_over_splits: int | None
    """The minimum number of classes across splits for classification tasks,
    None for regression tasks."""
    multiclass_max_n_classes_over_splits: int | None
    """The maximum number of classes across splits for classification tasks,
    None for regression tasks."""
    class_consistency_over_splits: bool | None
    """Whether the number of classes is consistent across splits for classification tasks,
    None for regression tasks."""

    num_instances: int
    """The total number of instances in the dataset."""
    num_features: int
    """The total number of features (excluding the target) in the dataset."""
    num_classes: int
    """The total number of classes in the dataset. -1 for regression tasks."""
    num_instance_groups: int
    """The total number of unique groups of data in the dataset. For normal IID data,
    this is equal to num_instances. For non-IID grouped data, this is equal to the number
    of groups in the data.
    """

    tabarena_task_name: str | None
    """The name of the task used for TabArena. This is used for better identification
    and can be set by the user."""
    task_id_str: str | None
    """The task ID string used for the task. This functions as the unique identifier
     of the source of the metadat. This can either be an OpenML task ID, or it is the
     identifier of a local task (see `UserTask.task_id_str`).
    """

    # -- Feature dtype flags (added later; default to None for backward compat) --
    has_datetime: bool | None = None
    """Whether the dataset contains datetime feature columns (incl. period dtypes)."""
    has_text: bool | None = None
    """Whether the dataset contains text (string) feature columns."""
    has_categorical: bool | None = None
    """Whether the dataset contains categorical feature columns."""
    has_numerical: bool | None = None
    """Whether the dataset contains numerical feature columns."""
    has_binary: bool | None = None
    """Whether the dataset contains any feature column with exactly two distinct
    non-null values."""
    has_high_cardinality_categorical: bool | None = None
    """Whether the dataset contains a categorical (category dtype) feature column
    with more than 50 unique values."""

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

    def has_supported_dtypes(self, *, required_dtypes: list[str] | None, forbidden_dtypes: list[str] | None) -> bool:
        """Check if the dataset contains only allowed dtypes based on the feature dtype flags."""
        flag_by_dtype = {
            "datetime": self.has_datetime,
            "text": self.has_text,
            "categorical": self.has_categorical,
            "numerical": self.has_numerical,
            "binary": self.has_binary,
            "high_cardinality_categorical": self.has_high_cardinality_categorical,
        }

        if required_dtypes is not None:
            for dtype in required_dtypes:
                if not flag_by_dtype.get(dtype):
                    return False

        if forbidden_dtypes is not None:
            for dtype in forbidden_dtypes:
                if flag_by_dtype.get(dtype):
                    return False

        return True

    def to_dict(self, *, exclude_splits_metadata: bool = False) -> dict:
        """Convert the task metadata to a dictionary for better visualization."""
        res = asdict(self)
        if exclude_splits_metadata:
            res.pop("splits_metadata")
        return res

    def to_dataframe(self, *, add_old_minimal_metadata: bool = False) -> pd.DataFrame:
        """Transform metadata to a DataFrame.

        If add_old_minimal_metadata is True, also add old minimal metadata for backward
        compatibility with old eval code. That is, we add the columns: "tid", "name", "task_type", "dataset",
        "n_samples_train_per_fold", "n_samples_test_per_fold".
        """
        rows = []
        static_metadata = self.to_dict(exclude_splits_metadata=True)
        for split_metadata in self.splits_metadata.values():
            rows.append(
                {
                    **static_metadata,
                    **split_metadata.to_dict(),
                }
            )

        df = pd.DataFrame(rows)

        # TODO: move somewhere else / get rid of this?
        if add_old_minimal_metadata:
            # Add old minimal metadata for backward compatibility with old eval code
            df["tid"] = int(self.task_id_str.split("|")[1])  # FIXME: avoid hacky way
            df["name"] = df["tabarena_task_name"]
            df["task_type"] = "classification"
            df.loc[~df["is_classification"], "task_type"] = "regression"
            df["dataset"] = df["tabarena_task_name"]
            df["n_samples_train_per_fold"] = df["num_instances_train"]
            df["n_samples_test_per_fold"] = df["num_instances_test"]

        return df

    @staticmethod
    def from_row(row: pd.Series) -> TabArenaTaskMetadata:
        """Reconstruct TabArenaTaskMetadata from a single dataframe row."""
        row_dict = row.to_dict()

        # Identify TabArenaTaskMetadata fields (excluding splits_metadata).
        # Fields with defaults are optional for backward compatibility with
        # older serialized metadata that may not contain newer columns.
        all_task_fields = {f.name for f in fields(TabArenaTaskMetadata) if f.name != "splits_metadata"}
        required_task_fields = {
            f.name
            for f in fields(TabArenaTaskMetadata)
            if f.name != "splits_metadata" and f.default is MISSING and f.default_factory is MISSING
        }
        if not all(name in row_dict for name in required_task_fields):
            raise ValueError(
                "Metadata row is missing required TabArenaTaskMetadata fields: "
                f"{required_task_fields - row_dict.keys()}"
            )
        task_kwargs = {key: row_dict[key] for key in all_task_fields if key in row_dict}

        # Identify SplitMetadata fields
        split_field_names = {f.name for f in fields(SplitMetadata)}
        if not all(name in row_dict for name in split_field_names):
            raise ValueError(
                f"Metadata row is missing required SplitMetadata fields: {split_field_names - row_dict.keys()}"
            )
        split_kwargs = {key: row_dict[key] for key in split_field_names if key in row_dict}
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
    num_instance_groups_train: int
    """The number unique groups of data in the train split.
    For normal IID data, this is always equal to `num_instances_train`.
    For non-IID grouped data, this is equal to the number of groups in the data.
    """
    num_instance_groups_test: int
    """The number unique groups of data in the test split."""
    num_classes_train: int
    """-1 for regression tasks, and maximal number of unique classes in the training
    set for classification tasks."""
    num_classes_test: int
    """Classes in test data"""
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
