"""A TaskWrapper backed by plain in-memory inputs — no OpenML objects involved.

``InMemoryTaskWrapper`` is the generic task implementation: a dataset frame, the
outer splits, and a ``TabArenaTaskMetadata`` fully define a runnable task. It is
what :meth:`UserTask.create_task` builds and what :meth:`UserTask.load` returns
for tasks persisted in the native format.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from tabarena.benchmark.task.wrapper import TaskWrapper

if TYPE_CHECKING:
    from collections.abc import Callable

    import pandas as pd

    from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, ValidationMetadata

#: The outer splits: ``{repeat: {fold: (train_indices, test_indices)}}`` with
#: positional (0-based) row indices into the dataset frame.
SplitsType = dict[int, dict[int, tuple[np.ndarray, np.ndarray]]]


class InMemoryTaskWrapper(TaskWrapper):
    """Runtime task defined by ``(dataset, splits, metadata)`` directly.

    Parameters
    ----------
    dataset: pd.DataFrame | Callable[[], pd.DataFrame]
        The full task frame *including* the target column, with a default
        ``RangeIndex`` and resolved dtypes — or a zero-argument loader returning
        it (the lazy-load hook: with ``lazy_load_data=True`` the loader is called
        per data access instead of keeping the frame resident, e.g. re-reading a
        local cache file).
    splits: SplitsType
        The outer evaluation splits (positional indices into ``dataset``).
    metadata: TabArenaTaskMetadata
        The task's metadata — required, since it carries the task definition
        (target, problem type, eval metric, split configuration, identity).
    lazy_load_data: bool, default False
        See ``TaskWrapper``; only useful when ``dataset`` is a loader.
    """

    def __init__(
        self,
        *,
        dataset: pd.DataFrame | Callable[[], pd.DataFrame],
        splits: SplitsType,
        metadata: TabArenaTaskMetadata,
        lazy_load_data: bool = False,
    ) -> None:
        assert metadata is not None, "InMemoryTaskWrapper requires a TabArenaTaskMetadata (it defines the task)."
        self._dataset_source = dataset
        self._splits = splits
        super().__init__(lazy_load_data=lazy_load_data, metadata=metadata)

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        dataset = self._dataset_source() if callable(self._dataset_source) else self._dataset_source
        target_name = self.metadata.target_name
        return dataset.drop(columns=[target_name]), dataset[target_name]

    @property
    def task_id(self) -> int:
        """The legacy integer ``tid``, parsed from the metadata's ``task_id_str``."""
        from tabarena.benchmark.task.metadata.schema import tid_from_task_id_str

        task_id_str = self.metadata.task_id_str
        if task_id_str is None:
            raise ValueError("This task's metadata carries no `task_id_str`; no integer task id is available.")
        return tid_from_task_id_str(task_id_str)

    def get_split_dimensions(self) -> tuple[int, int, int]:
        n_repeats = len(self._splits)
        n_folds = len(next(iter(self._splits.values())))
        return n_repeats, n_folds, 1

    def get_split_indices(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        assert sample == 0, "In-memory tasks do not support samples (learning curves)."
        train_indices, test_indices = self._splits[repeat][fold]
        return np.asarray(train_indices, dtype=int), np.asarray(test_indices, dtype=int)

    def get_validation_metadata(self) -> ValidationMetadata:
        """Projected from the task's metadata (which carries the split configuration)."""
        return self.metadata.to_validation_metadata()
