from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields, replace
from enum import StrEnum
from typing import Annotated, Literal

import pandas as pd
from loguru import logger

SplitIndex = Annotated[str, "format: r{int}f{int}"]

SplitTimeHorizonTypes = str | int | float
SplitTimeHorizonUnitTypes = Literal["steps", "days", "weeks", "months", "years"] | str


class GroupLabelTypes(StrEnum):
    PER_SAMPLE = "per_sample"
    PER_GROUP = "per_group"


def derive_task_type(*, time_on: str | None, group_on: str | list[str] | None) -> str:
    """Classify a task's split regime from its split columns.

    Returns ``"temporal"`` if ``time_on`` is set, ``"grouped"`` if ``group_on`` is set,
    else ``"random"`` (IID). Mirrors the Data Foundry warehouse ``task_type``.
    """
    if time_on is not None:
        return "temporal"
    if group_on is not None:
        return "grouped"
    return "random"


def tid_from_task_id_str(task_id_str: str | int) -> int:
    """Parse the legacy integer ``tid`` from a ``task_id_str``.

    Serialized spec ids follow the ``"{Prefix}|{task_id}|..."`` convention (see
    :attr:`~tabarena.benchmark.task.spec.TaskSpec.task_id_str`), so the embedded
    ``{task_id}`` segment is returned for any prefixed form — no spec reconstruction
    needed; a plain OpenML integer task id (str or int) parses as-is.
    """
    s = str(task_id_str)
    parts = s.split("|")
    return int(parts[1]) if len(parts) > 1 else int(s)


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

    data_foundry_uri: str | None = None
    """For tasks sourced from a Data Foundry collection, the collection-entry relative
    path (``<unique_name>/[versions/]<uuid>``) that identifies the source container.
    Used to (re)download and materialize the task on demand. ``None`` for tasks that do
    not originate from Data Foundry."""

    # -- Warehouse-level metadata (added later; default to None for backward compat) --
    # These align the in-task metadata with the columns previously merged in from a
    # separate warehouse_metadata.csv. Dataset-derived fields
    # are computed at creation when the dataset is available (data_foundry / OpenML
    # task path); they remain None when only tabular metadata is known (e.g. v0.1).
    task_type: str | None = None
    """The split regime of the task: ``"random"`` (IID), ``"temporal"``, or
    ``"grouped"``. Derived from ``time_on`` / ``group_on`` (see :func:`derive_task_type`)."""
    num_text_cols: int | None = None
    """Number of text (string-dtype) feature columns."""
    num_high_cardinality_cats: int | None = None
    """Number of categorical (category-dtype) feature columns with more than 50 unique values."""
    num_cols_after_preprocessing: int | None = None
    """Estimated feature count after preprocessing: numerical_non_binary +
    categorical_non_binary + num_text * 32 + num_binary + num_datetime * 10
    (matches the Data Foundry warehouse computation)."""
    missing_value_fraction: float | None = None
    """Fraction of missing values across all feature cells (excluding target/group cols)."""
    domain: str | None = None
    """Application domain of the dataset (e.g. ``"medical & healthcare"``)."""
    dataset_year: str | None = None
    """Year the dataset originates from."""
    source: str | None = None
    """Origin of the dataset (e.g. ``"Kaggle"``, ``"OpenML"``, ``"UCI"``)."""

    def to_validation_metadata(self) -> ValidationMetadata:
        """Project this task metadata onto a ``ValidationMetadata`` (the split-config subset).

        Single source for the task -> validation-metadata mapping; the fold-count policy
        fields keep their defaults.
        """
        return ValidationMetadata.from_task_metadata(self)

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
                "This is only supported for tasks with exactly one split.",
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
        compatibility with old eval code. That is, we add the columns: "tid", "name",
        "dataset", "n_samples_train_per_fold", "n_samples_test_per_fold".
        """
        rows = []
        static_metadata = self.to_dict(exclude_splits_metadata=True)
        for split_metadata in self.splits_metadata.values():
            rows.append(
                {
                    **static_metadata,
                    **split_metadata.to_dict(),
                },
            )

        df = pd.DataFrame(rows)

        # TODO: move somewhere else / get rid of this?
        if add_old_minimal_metadata:
            # Add old minimal metadata for backward compatibility with old eval code
            # Integer task id: the UserTask hash id (``UserTask|<id>|...``) for local
            # tasks, or the plain OpenML integer task id otherwise (handles str or int).
            df["tid"] = tid_from_task_id_str(self.task_id_str)
            df["name"] = df["tabarena_task_name"]
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
                f"{required_task_fields - row_dict.keys()}",
            )
        task_kwargs = {key: row_dict[key] for key in all_task_fields if key in row_dict}

        # The DataFrame/CSV round-trip serializes None as NaN (see ``to_dataframe``);
        # map scalar NA back to None for the fields that permit it, so e.g. an unset
        # ``group_on`` does not surface as ``float("nan")`` downstream.
        optional_task_fields = {f.name for f in fields(TabArenaTaskMetadata) if "None" in str(f.type)}
        for key in optional_task_fields & task_kwargs.keys():
            value = task_kwargs[key]
            if isinstance(value, float) and pd.isna(value):
                task_kwargs[key] = None

        # Identify SplitMetadata fields
        split_field_names = {f.name for f in fields(SplitMetadata)}
        if not all(name in row_dict for name in split_field_names):
            raise ValueError(
                f"Metadata row is missing required SplitMetadata fields: {split_field_names - row_dict.keys()}",
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


@dataclass(frozen=True)
class ValidationMetadata:
    """Task-derived metadata (and policy) for building the validation split.

    Projected from a task's split metadata (see ``TabArenaTaskMetadata.to_validation_metadata``
    / ``ValidationMetadata.from_task_metadata``) and consumed by the AutoGluon wrappers via
    ``tabarena.benchmark.exec_models.autogluon_utils.resolve_validation_splits``.

    The split-column fields mirror the task metadata; ``target_name`` is carried so the
    wrapper can name its internal label column (the splitting logic itself does not use it).
    The ``*_num_folds`` / ``*_num_repeats`` / ``max_samples_for_tiny_data`` fields encode the
    fold-count policy resolved in ``resolve_number_of_splits``. Whether this metadata is
    applied at all is a separate decision the wrapper makes (``use_task_specific_validation``).
    """

    target_name: str | None = None
    """Name of the target column (used by the wrapper to name its label column)."""
    stratify_on: str | None = None
    """Column to stratify validation splits on."""
    group_on: str | list[str] | None = None
    """Column(s) identifying groups for group-wise validation splitting."""
    time_on: str | None = None
    """Column identifying time for time-based validation splitting."""
    group_time_on: str | None = None
    """Column identifying time within groups."""
    group_labels: GroupLabelTypes | None = None
    """Whether ``group_on`` carries labels per sample or per group."""
    split_time_horizon: SplitTimeHorizonTypes | None = None
    """Time horizon for the deployment/test data."""
    split_time_horizon_unit: SplitTimeHorizonUnitTypes | None = None
    """Unit for ``split_time_horizon`` (e.g. days, months, years)."""

    # Fold-count policy. Above ``max_samples_for_tiny_data`` (group) instances we expect the
    # benchmark defaults; at or below it we switch to the denser tiny-data regime (more
    # folds/repeats) for a more reliable validation score. See ``resolve_number_of_splits``.
    default_num_folds: int = 8
    """Expected number of folds for non-tiny datasets."""
    default_num_repeats: int = 1
    """Expected number of repeats for non-tiny datasets (``None`` is also allowed)."""
    tiny_data_num_folds: int = 5
    """Number of folds to use for tiny datasets."""
    tiny_data_num_repeats: int = 5
    """Number of repeats to use for tiny datasets."""
    max_samples_for_tiny_data: int = 500
    """At or below this many (group) instances, the tiny-data regime applies."""

    @classmethod
    def from_task_metadata(cls, metadata) -> ValidationMetadata:
        """Project a task-metadata-like object onto a ``ValidationMetadata``.

        ``metadata`` is any object exposing the split-metadata attributes (a
        ``TabArenaTaskMetadata`` or a TabArena OpenML task); the fold-count policy fields
        keep their defaults.
        """
        return cls(
            target_name=metadata.target_name,
            stratify_on=metadata.stratify_on,
            group_on=metadata.group_on,
            time_on=metadata.time_on,
            group_time_on=metadata.group_time_on,
            group_labels=metadata.group_labels,
            split_time_horizon=metadata.split_time_horizon,
            split_time_horizon_unit=metadata.split_time_horizon_unit,
        )

    @classmethod
    def from_config(
        cls,
        config: ValidationMetadata | dict | None,
        *,
        base: ValidationMetadata | None = None,
    ) -> ValidationMetadata:
        """Build metadata from ``config``, layering it over ``base``.

        ``config`` may be:

        - ``None`` — use ``base`` unchanged.
        - a ``dict`` — override the corresponding ``base`` fields individually (keys absent
          from the dict keep their ``base`` value).
        - a ``ValidationMetadata`` — used as-is (a complete, explicit spec).

        ``base`` defaults to a fresh ``ValidationMetadata()``; pass the task-derived metadata
        to let per-key dict overrides fall back to the task's values.
        """
        base = base if base is not None else cls()
        if config is None:
            return base
        if isinstance(config, ValidationMetadata):
            return config
        return replace(base, **config)

    def resolve_number_of_splits(
        self,
        *,
        num_folds: int,
        num_repeats: int | None,
        num_group_instances: int,
    ) -> tuple[int, int | None]:
        """Resolve the (folds, repeats) to use given the data size.

        At or below ``max_samples_for_tiny_data`` (group) instances, switch to the tiny-data
        regime (``tiny_data_num_folds`` / ``tiny_data_num_repeats``); otherwise assert and
        return the configured defaults (``default_num_folds`` / ``default_num_repeats``).

        Parameters
        ----------
        num_folds: int
            The number of folds entered for validation.
        num_repeats: int
            The number of repeats entered for validation.
        num_group_instances: int
            The number of group instances in the data.
        """
        if num_group_instances <= self.max_samples_for_tiny_data:
            logger.info(
                f"\nTiny data ({num_group_instances} <= {self.max_samples_for_tiny_data}): using "
                f"num_bag_folds={self.tiny_data_num_folds}, num_bag_sets={self.tiny_data_num_repeats}.",
            )
            return self.tiny_data_num_folds, self.tiny_data_num_repeats

        # Larger data: expect (and keep) the configured benchmark defaults.
        assert num_folds == self.default_num_folds
        assert num_repeats in (self.default_num_repeats, None)
        return num_folds, num_repeats


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


def to_legacy_task_metadata(task_metadata: list[TabArenaTaskMetadata]) -> pd.DataFrame:
    """Convert a list of :class:`TabArenaTaskMetadata` into the legacy ``task_metadata``
    DataFrame consumed by ``TabArenaContext`` and ``ExperimentBatchRunner``.

    The legacy format is **one row per dataset** with the columns those consumers read:

    * ``dataset`` / ``tid`` — dataset identity (``ExperimentBatchRunner`` builds its
      dataset→tid map from these; both consumers ``drop_duplicates(subset="dataset")``).
    * ``n_folds`` / ``n_repeats`` — *aggregate* per-dataset split counts, used to expand
      the (dataset, fold, repeat) grid.
    * ``problem_type``, ``n_features``, ``n_classes``, ``n_samples_train_per_fold`` — the
      columns the subset predicates reference. ``n_samples_train_per_fold`` is aliased to
      ``max_train_rows`` when the legacy frame is expanded into the subset task grid
      (``compare._task_grid_from_legacy_df``) for the size buckets.

    Two things make this a non-trivial mapping (see :func:`TabArenaTaskMetadata.to_dataframe`):

    * ``to_dataframe`` emits one row *per split* and never the aggregate ``n_folds`` /
      ``n_repeats``, so we group by dataset and count distinct folds/repeats. Collapsing to
      one row per dataset is required: ``subset_results`` merges this on ``"dataset"``, and
      a multi-row-per-dataset frame would fan that merge out.
    * the predicate columns ``n_features`` / ``n_classes`` are named ``num_features`` /
      ``num_classes`` on the schema and must be renamed (the predicates raise ``KeyError``
      on missing columns rather than skipping them).

    Requires each task to carry a parseable ``task_id_str`` (enforced upstream by
    :class:`~tabarena.benchmark.task.metadata.collection.TaskMetadataCollection`); an
    empty list yields an empty DataFrame.
    """
    if not task_metadata:
        return pd.DataFrame()

    per_split = pd.concat(
        [ttm.to_dataframe(add_old_minimal_metadata=True) for ttm in task_metadata],
        ignore_index=True,
    )

    # Aggregate the split grain up to the dataset grain the legacy format expects.
    # n_samples_*_per_fold is the mean per-fold size, kept as a float: it matches the curated
    # per-fold average and is the exact inverse of from_legacy_df (which stamps one per-fold
    # value onto every split), so the conversion round-trips without max()/int() loss.
    # (NB: the subset predicates alias n_samples_train_per_fold to ``max_train_rows``.)
    aggregates = per_split.groupby("dataset", sort=False).agg(
        n_folds=("fold", "nunique"),
        n_repeats=("repeat", "nunique"),
        n_samples_train_per_fold=("num_instances_train", "mean"),
        n_samples_test_per_fold=("num_instances_test", "mean"),
    )

    # Keep one (dataset-constant) static row per dataset, drop the now-meaningless
    # split-grain columns, and splice the dataset-level aggregates back in.
    split_cols = {f.name for f in fields(SplitMetadata)} | {"split_index"}
    static = (
        per_split.drop_duplicates(subset="dataset")
        .set_index("dataset")
        .drop(columns=list(split_cols | set(aggregates.columns)), errors="ignore")
    )

    legacy = static.join(aggregates).rename(
        columns={"num_features": "n_features", "num_classes": "n_classes"},
    )
    return legacy.reset_index()
