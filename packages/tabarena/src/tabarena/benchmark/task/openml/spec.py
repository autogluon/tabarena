"""TaskSpec for published OpenML tasks (vended by downloading via task id)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.task.spec import TaskSpec

if TYPE_CHECKING:
    from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper
    from tabarena.benchmark.task.wrapper import TaskWrapper


class OpenMLTaskSpec(TaskSpec):
    """A task identified by an integer OpenML task id, loaded from the OpenML server.

    ``cache_key`` is the bare integer id — the historical results-cache key for
    OpenML tasks — and ``task_id_str`` its string form.
    """

    def __init__(self, task_id: int) -> None:
        self._task_id = int(task_id)

    @property
    def task_id(self) -> int:
        return self._task_id

    @property
    def task_id_str(self) -> str:
        return str(self._task_id)

    @property
    def cache_key(self) -> int:
        return self._task_id

    def load(self) -> OpenMLTaskWrapper:
        from tabarena.benchmark.task.openml.task_wrapper import OpenMLTaskWrapper

        return OpenMLTaskWrapper.from_task_id(task_id=self._task_id, metadata=self.task_metadata)

    def resolve_task_name(self, task: TaskWrapper) -> str:
        """The OpenML dataset name (available on the loaded task's dataset object)."""
        return task.task.get_dataset().name

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self._task_id})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, OpenMLTaskSpec) and other._task_id == self._task_id

    def __hash__(self) -> int:
        return hash((type(self).__name__, self._task_id))
