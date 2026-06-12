from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Self

from autogluon.common.savers import save_json
from openml.tasks.task import OpenMLSupervisedTask

from tabarena.benchmark.task.wrapper import TaskWrapper

from .task_utils import get_ag_problem_type, get_task_data, get_task_with_retry

if TYPE_CHECKING:
    import numpy as np
    import pandas as pd

    from tabarena.benchmark.task.metadata import TabArenaTaskMetadata, ValidationMetadata

logger = logging.getLogger(__name__)


class OpenMLTaskWrapper(TaskWrapper):
    """:class:`TaskWrapper` backed by a (live or local) ``OpenMLSupervisedTask``.

    Data and split indices are delegated to the OpenML task object; the problem
    type and target come from the task definition. Use :meth:`from_task_id` to
    vend a task from the OpenML server.
    """

    def __init__(
        self,
        task: OpenMLSupervisedTask,
        *,
        use_task_eval_metric: bool = False,
        lazy_load_data: bool = False,
        metadata: TabArenaTaskMetadata | None = None,
    ):
        assert isinstance(task, OpenMLSupervisedTask)
        self.task: OpenMLSupervisedTask = task
        super().__init__(lazy_load_data=lazy_load_data, metadata=metadata)

        if metadata is None:
            self.problem_type = get_ag_problem_type(self.task)
            self.label = self.task.target_name

            # TODO: check if we can always use the eval metric from the task for TabArena
            #   tasks?
            if use_task_eval_metric and (self.task.evaluation_measure is not None):
                self._eval_metric = self.task.evaluation_measure
            else:
                self._eval_metric = None
        else:
            # Metadata is the source of truth (set by the base); fail fast if it
            # contradicts the task definition itself — that means the collection and
            # the loaded task drifted apart.
            assert self.label == self.task.target_name, (
                f"Task metadata target_name={self.label!r} contradicts the loaded OpenML task's "
                f"target_name={self.task.target_name!r}."
            )
            task_problem_type = get_ag_problem_type(self.task)
            assert self.problem_type == task_problem_type, (
                f"Task metadata problem_type={self.problem_type!r} contradicts the loaded OpenML task's "
                f"problem_type={task_problem_type!r}."
            )
            if use_task_eval_metric and self._eval_metric is None and self.task.evaluation_measure is not None:
                self._eval_metric = self.task.evaluation_measure

    @classmethod
    def from_task_id(cls, task_id: int, *, metadata: TabArenaTaskMetadata | None = None) -> Self:
        task = get_task_with_retry(task_id=task_id)
        return cls(task, metadata=metadata)

    def _load_data(self) -> tuple[pd.DataFrame, pd.Series]:
        return get_task_data(task=self.task)

    @property
    def task_id(self) -> int:
        return self.task.task_id

    @property
    def dataset_id(self) -> int:
        return self.task.dataset_id

    @property
    def dataset_name(self) -> str | None:
        if self.metadata is not None:
            return self.metadata.dataset_name
        return self.task.get_dataset().name

    def get_split_dimensions(self) -> tuple[int, int, int]:
        n_repeats, n_folds, n_samples = self.task.get_split_dimensions()
        return n_repeats, n_folds, n_samples

    def get_split_indices(self, fold: int = 0, repeat: int = 0, sample: int = 0) -> tuple[np.ndarray, np.ndarray]:
        train_indices, test_indices = self.task.get_train_test_split_indices(fold=fold, repeat=repeat, sample=sample)
        return train_indices, test_indices

    def save_metadata(self, path: str):
        metadata = dict(
            label=self.label,
            problem_type=self.problem_type,
            num_rows=self._n_rows,
            num_cols=self._n_cols,
            task_id=self.task.task_id,
            dataset_id=self.task.dataset_id,
            openml_url=self.task.openml_url,
        )
        path_full = f"{path}metadata.json"
        save_json.save(path=path_full, obj=metadata)

    def get_validation_metadata(self) -> ValidationMetadata:
        """Task-derived validation-split metadata.

        Pulls the split metadata from a ``TabArenaOpenMLSupervisedTask`` (group/time/stratify
        columns, time horizon); a plain ``OpenMLSupervisedTask`` yields the target name only.
        Whether the metadata is applied is decided by the wrapper (the experiment runner
        enables it; see ``Experiment.task_cache_scope``).
        """
        from tabarena.benchmark.task.metadata import ValidationMetadata
        from tabarena.benchmark.task.openml.metadata_mixin import TabArenaOpenMLSupervisedTask

        oml_task: TabArenaOpenMLSupervisedTask | OpenMLSupervisedTask = self.task

        if not isinstance(oml_task, TabArenaOpenMLSupervisedTask):
            return ValidationMetadata(target_name=oml_task.target_name)

        # TabArena tasks carry the split-metadata attributes; project them in one place.
        return ValidationMetadata.from_task_metadata(oml_task)
