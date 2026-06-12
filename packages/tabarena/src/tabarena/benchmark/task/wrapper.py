"""Source-agnostic runtime interface of a benchmark task.

``TaskWrapper`` is the type the run engine consumes (``ExperimentRunner``,
``Experiment.run``): it provides the data splits, problem metadata, and the
split-index arithmetic for one task, independent of where the task comes from.
Everything task-source-specific (how the data and split indices are obtained)
is deferred to subclasses — e.g. :class:`~tabarena.benchmark.task.openml.OpenMLTaskWrapper`
for OpenML-backed tasks.
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd
from autogluon.common.savers import save_pd
from autogluon.core.utils import generate_train_test_split

from tabarena.benchmark.task.utils import get_split_idx, get_split_vals_from_split_idx

if TYPE_CHECKING:
    import numpy as np

    from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, ValidationMetadata

logger = logging.getLogger(__name__)

#: Fields skipped by :meth:`TaskWrapper.validate_metadata` by default: identity /
#: provenance values that are assigned, not derivable from the loaded data.
DEFAULT_VALIDATE_METADATA_IGNORED_FIELDS = frozenset(
    {
        "tabarena_task_name",
        "task_id_str",
        "dataset_name",
        "data_foundry_uri",
        "domain",
        "dataset_year",
        "source",
    },
)


class TaskWrapper(ABC):
    """Runtime interface of a benchmark task, as consumed by the run engine.

    A subclass owns the task's *source* (OpenML object, local files, in-memory
    frames, ...) and must:

    * set ``self.problem_type`` (AutoGluon convention: ``"binary"`` /
      ``"multiclass"`` / ``"regression"``), ``self.label`` (target column name),
      and optionally ``self._eval_metric`` (else the per-problem-type default of
      :attr:`eval_metric` applies) — unless a ``metadata`` is passed, which sets
      all three (the metadata is then the single source of truth);
    * implement the data hook (:meth:`_load_data`) and the split hooks
      (:meth:`get_split_dimensions`, :meth:`get_split_indices`);
    * expose its identity via :attr:`task_id`.

    The base class owns everything source-agnostic: split-index arithmetic,
    train/test split assembly (with optional lazy data loading and subsampling),
    metric-error computation, the light dtype flags used for task handling, and
    the metadata round-trip (:meth:`compute_metadata` / :meth:`validate_metadata`).
    """

    problem_type: str
    label: str
    _eval_metric: str | None = None

    def __init__(self, *, lazy_load_data: bool = False, metadata: TabArenaTaskMetadata | None = None) -> None:
        """Load the data once to derive shape/dtype metadata; keep it unless lazy.

        Parameters
        ----------
        lazy_load_data: bool, default False
            If True, the data is dropped after deriving the light metadata and
            re-loaded on demand via :meth:`_load_data` (one load per access);
            ``self.X`` / ``self.y`` are then not available as attributes.
        metadata: TabArenaTaskMetadata | None, default None
            The task's metadata, when known up front (e.g. from the run's
            ``TaskMetadataCollection``). When given it is the single source of
            truth: ``problem_type``, ``label``, ``eval_metric``, and the dtype
            flags are read from it instead of being re-derived from the source,
            so the run cannot diverge from the collection that scheduled it.
            Verify a stored metadata against the loaded data with
            :meth:`validate_metadata`.
        """
        self.metadata = metadata
        self.lazy_load_data = lazy_load_data
        X, y = self._load_data()
        self._n_rows, self._n_cols = X.shape

        # Light metadata for task handling.
        self._has_datetime = len(X.select_dtypes(include=["datetime64"]).columns) > 0
        self._has_text = len(X.select_dtypes(include=["string"]).columns) > 0
        self._has_categorical = len(X.select_dtypes(include=["category"]).columns) > 0
        self._has_numeric = len(X.select_dtypes(include=["number"]).columns) > 0

        if not self.lazy_load_data:
            self.X, self.y = X, y

        if metadata is not None:
            self.problem_type = metadata.problem_type
            self.label = metadata.target_name
            if metadata.eval_metric is not None:
                self._eval_metric = metadata.eval_metric

    # --- Source-specific hooks ----------------------------------------------------------
    @abstractmethod
    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        """Load and return the full ``(X, y)`` of the task from its source."""

    @property
    @abstractmethod
    def task_id(self) -> int:
        """Integer identifier of the task (recorded as ``tid`` on results)."""

    @abstractmethod
    def get_split_dimensions(self) -> tuple[int, int, int]:
        """Return the task's ``(n_repeats, n_folds, n_samples)``."""

    @abstractmethod
    def get_split_indices(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Return the ``(train_indices, test_indices)`` of one outer split."""

    def get_validation_metadata(self) -> ValidationMetadata:
        """Task-derived validation-split metadata (target name only by default).

        Subclasses whose source carries split metadata (group/time/stratify columns,
        time horizon) override this to project it; whether the metadata is applied is
        decided by the caller (see ``Experiment.task_cache_scope``).
        """
        from tabarena.benchmark.task.metadata import ValidationMetadata

        return ValidationMetadata(target_name=self.label)

    # --- Problem metadata ----------------------------------------------------------------
    @property
    def has_text(self) -> bool:
        """Whether the task's features contain text (string-dtype) columns.

        Metadata-first: when a :attr:`metadata` carries the flag, its (binary-aware)
        definition wins over the wrapper's light dtype scan.
        """
        if self.metadata is not None and self.metadata.has_text is not None:
            return self.metadata.has_text
        return self._has_text

    @property
    def dataset_name(self) -> str | None:
        """Simple name of the dataset behind this task (``None`` when unknown)."""
        if self.metadata is not None:
            return self.metadata.dataset_name
        return None

    @property
    def eval_metric(self) -> str:
        if self._eval_metric is not None:
            return self._eval_metric
        from tabarena.benchmark.task.metrics import default_eval_metric

        return default_eval_metric(self.problem_type)

    def compute_error(self, y_true, y_pred) -> float:
        eval_metric = self.eval_metric
        from autogluon.core.metrics import get_metric

        scorer = get_metric(metric=eval_metric, problem_type=self.problem_type)
        return scorer.error(y_true, y_pred)

    # --- Metadata round-trip ---------------------------------------------------------------
    def compute_metadata(
        self,
        *,
        tabarena_task_name: str | None = None,
        task_id_str: str | None = None,
    ) -> TabArenaTaskMetadata:
        """Compute this task's exact ``TabArenaTaskMetadata`` from its loaded data and splits.

        Available on every task wrapper out of the box: the data, the outer splits,
        and the split configuration (via :meth:`get_validation_metadata`) are projected
        onto :func:`~tabarena.benchmark.task.metadata.compute.compute_task_metadata`
        — the same single implementation the local-task creation path uses, so a
        computed and a stored metadata can only differ in *values* (stale entries),
        never in structure or definition.

        ``tabarena_task_name`` / ``task_id_str`` default to the attached
        :attr:`metadata`'s identity (when present); the recorded ``eval_metric`` is
        the task's explicit metric (``None`` when the per-problem-type default
        applies — mirroring how stored metadata records it).
        """
        from tabarena.benchmark.task.metadata.compute import compute_task_metadata

        n_repeats, n_folds, n_samples = self.get_split_dimensions()
        assert n_samples == 1, "Only one sample per split is supported so far!."
        splits = {
            repeat: {fold: self.get_split_indices(fold=fold, repeat=repeat) for fold in range(n_folds)}
            for repeat in range(n_repeats)
        }

        validation_metadata = self.get_validation_metadata()
        if tabarena_task_name is None and self.metadata is not None:
            tabarena_task_name = self.metadata.tabarena_task_name
        if task_id_str is None and self.metadata is not None:
            task_id_str = self.metadata.task_id_str

        return compute_task_metadata(
            dataset=self.combine_X_y(),
            dataset_name=self.dataset_name,
            target_name=self.label,
            is_classification=self.problem_type in ("binary", "multiclass"),
            eval_metric=self._eval_metric,
            splits=splits,
            stratify_on=validation_metadata.stratify_on,
            group_on=validation_metadata.group_on,
            time_on=validation_metadata.time_on,
            group_time_on=validation_metadata.group_time_on,
            group_labels=validation_metadata.group_labels,
            split_time_horizon=validation_metadata.split_time_horizon,
            split_time_horizon_unit=validation_metadata.split_time_horizon_unit,
            tabarena_task_name=tabarena_task_name,
            task_id_str=task_id_str,
        )

    def validate_metadata(
        self,
        expected: TabArenaTaskMetadata | None = None,
        *,
        ignore_fields: set[str] | None = None,
    ) -> TabArenaTaskMetadata:
        """Recompute the metadata from the loaded task and assert it matches ``expected``.

        The equivalence check between a *stored* task metadata (a collection entry)
        and the task as actually loaded: every field of :meth:`compute_metadata`'s
        result must equal ``expected``'s (default: the attached :attr:`metadata`),
        except the assigned identity/provenance fields in
        :data:`DEFAULT_VALIDATE_METADATA_IGNORED_FIELDS` (or an explicit
        ``ignore_fields``). Raises ``AssertionError`` listing every diverging field;
        returns the computed metadata on success.

        Note: stored metadata may legitimately carry split-configuration fields
        (``stratify_on``, ...) that a *plain* downloaded OpenML task object does not
        expose — pass those in ``ignore_fields`` when validating such tasks.
        """
        if expected is None:
            expected = self.metadata
        if expected is None:
            raise ValueError("No metadata to validate against: pass `expected` or construct the wrapper with one.")
        ignored = DEFAULT_VALIDATE_METADATA_IGNORED_FIELDS if ignore_fields is None else ignore_fields

        computed = self.compute_metadata()
        computed_dict = computed.to_dict()
        expected_dict = expected.to_dict()
        diffs = [
            f"\t{field}: expected={expected_dict[field]!r}, computed={computed_dict[field]!r}"
            for field in computed_dict
            if field not in ignored and expected_dict[field] != computed_dict[field]
        ]
        if diffs:
            raise AssertionError(
                f"Task metadata mismatch for {expected.tabarena_task_name or expected.dataset_name!r} "
                f"({len(diffs)} field(s)):\n" + "\n".join(diffs),
            )
        return computed

    # --- Split-index arithmetic ----------------------------------------------------------
    def get_split_idx(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> int:
        n_repeats, n_folds, n_samples = self.get_split_dimensions()
        return get_split_idx(
            fold=fold,
            repeat=repeat,
            sample=sample,
            n_folds=n_folds,
            n_repeats=n_repeats,
            n_samples=n_samples,
        )

    def get_split_vals_from_split_idx(self, split_idx: int) -> tuple[int, int, int]:
        n_repeats, n_folds, n_samples = self.get_split_dimensions()
        return get_split_vals_from_split_idx(
            split_idx=split_idx,
            n_folds=n_folds,
            n_repeats=n_repeats,
            n_samples=n_samples,
        )

    # --- Data access ---------------------------------------------------------------------
    def get_X_y(self) -> tuple[pd.DataFrame, pd.Series]:
        """The full ``(X, y)`` of the task, loading on demand when data is lazy-loaded."""
        if self.lazy_load_data:
            return self._load_data()
        return self.X, self.y

    def combine_X_y(self) -> pd.DataFrame:
        X, y = self.get_X_y()
        return pd.concat([X, y.to_frame(name=self.label)], axis=1)

    def save_data(self, path: str, file_type=".csv", train_indices=None, test_indices=None):
        data = self.combine_X_y()
        if train_indices is not None and test_indices is not None:
            train_data = data.loc[train_indices]
            test_data = data.loc[test_indices]
            save_pd.save(f"{path}train{file_type}", train_data)
            save_pd.save(f"{path}test{file_type}", test_data)
        else:
            save_pd.save(f"{path}data{file_type}", data)

    def get_train_test_split(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float | None = None,
        test_size: int | float | None = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
        if train_indices is None or test_indices is None:
            train_indices, test_indices = self.get_split_indices(fold=fold, repeat=repeat, sample=sample)

        if self.lazy_load_data:
            X, y = self._load_data()
            X_train = X.loc[train_indices].copy()
            y_train = y[train_indices].copy()
            X_test = X.loc[test_indices].copy()
            y_test = y[test_indices].copy()
            del X, y
        else:
            X, y = self.X, self.y
            X_train = X.loc[train_indices]
            y_train = y[train_indices]
            X_test = X.loc[test_indices]
            y_test = y[test_indices]

        if train_size is not None:
            X_train, y_train = self.subsample(X=X_train, y=y_train, size=train_size, random_state=random_state)
        if test_size is not None:
            X_test, y_test = self.subsample(X=X_test, y=y_test, size=test_size, random_state=random_state)

        return X_train, y_train, X_test, y_test

    def get_train_test_split_combined(
        self,
        fold: int = 0,
        repeat: int = 0,
        sample: int = 0,
        train_indices: np.ndarray = None,
        test_indices: np.ndarray = None,
        train_size: int | float | None = None,
        test_size: int | float | None = None,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        X_train, y_train, X_test, y_test = self.get_train_test_split(
            fold=fold,
            repeat=repeat,
            sample=sample,
            train_indices=train_indices,
            test_indices=test_indices,
            train_size=train_size,
            test_size=test_size,
            random_state=random_state,
        )
        train_data = pd.concat([X_train, y_train.to_frame(name=self.label)], axis=1)
        test_data = pd.concat([X_test, y_test.to_frame(name=self.label)], axis=1)
        return train_data, test_data

    @classmethod
    def to_csv_format(cls, X: pd.DataFrame) -> pd.DataFrame:
        """Converts X to the dtypes that it would have if it were saved to a CSV and then loaded."""
        s_buf = io.StringIO()
        X_index = X.index
        X.to_csv(s_buf, index=False)
        s_buf.seek(0)
        X = pd.read_csv(s_buf, low_memory=False)
        X.index = X_index
        return X

    def subsample(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        size: int | float,
        random_state: int = 0,
    ) -> tuple[pd.DataFrame, pd.Series]:
        if isinstance(size, int) and size >= len(X):
            return X, y
        if isinstance(size, float) and size >= 1:
            return X, y
        X, _, y, _ = generate_train_test_split(
            X=X,
            y=y,
            problem_type=self.problem_type,
            train_size=size,
            random_state=random_state,
        )
        return X, y

    def subsample_combined(
        self,
        data: pd.DataFrame,
        size: int | float,
        random_state: int = 0,
    ) -> pd.DataFrame:
        data, _ = self.subsample(X=data, y=data[self.label], size=size, random_state=random_state)
        return data
