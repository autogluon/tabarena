"""Base source for OpenML-task-backed metadata (integer OpenML task ids).

Sources whose tasks are identified by integer OpenML task ids (e.g. TabArena v0.1)
can inherit :meth:`OpenMLTaskMetadataSource.materialize` to pre-cache each task's
dataset + splits into the OpenML cache. This warms a (shared) cache up front so
SLURM workers don't each re-download — replacing the old manual download step.
``load`` stays abstract: subclasses decide where the metadata table comes from.

Cache location: tasks are cached into OpenML's root cache directory. By default
that is whatever the ambient ``openml.config`` is set to — in the SLURM pipeline
this is configured by ``PathSetup.ensure_runtime_dirs`` (which honors the run's
``openml_cache``) before ``load_task_metadata`` runs, and otherwise falls back to
``~/.cache/openml``. Pass ``openml_cache_dir`` to override it explicitly (mirrors
the old pre-download ``--directory`` flag).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.task.metadata.sources.base import TaskMetadataSource

if TYPE_CHECKING:
    from pathlib import Path

    from tabarena.benchmark.task.metadata.schema import TabArenaTaskMetadata


class OpenMLTaskMetadataSource(TaskMetadataSource):
    """Source whose tasks are OpenML tasks; ``materialize`` pre-caches them."""

    def __init__(self, *, openml_cache_dir: str | Path | None = None) -> None:
        """Initialize the source.

        Args:
            openml_cache_dir: Optional OpenML root cache directory to cache tasks
                into. When set, it is applied globally via
                ``openml.config.set_root_cache_directory`` before downloading
                (mirroring the old pre-download step). When ``None``, the ambient
                OpenML cache is used — in the SLURM pipeline this is configured by
                ``PathSetup.ensure_runtime_dirs`` (else ``~/.cache/openml``).
        """
        self.openml_cache_dir = openml_cache_dir

    def materialize(self, task_metadata: list[TabArenaTaskMetadata]) -> None:
        """Download each OpenML task's dataset + splits into the OpenML cache, in place.

        Tasks whose ``task_id_str`` is not an integer OpenML id are skipped. Task ids
        are de-duplicated (splits of the same task share one id), so each dataset is
        fetched once.
        """
        task_ids: list[int] = []
        for ttm in task_metadata:
            try:
                task_ids.append(int(ttm.task_id_str))
            except (TypeError, ValueError):
                continue  # not an OpenML integer task id (e.g. a local UserTask) — skip
        unique_task_ids = list(dict.fromkeys(task_ids))
        if not unique_task_ids:
            return

        import openml
        from tqdm import tqdm

        if self.openml_cache_dir is not None:
            openml.config.set_root_cache_directory(root_cache_directory=str(self.openml_cache_dir))

        cache_dir = openml.config.get_cache_directory()
        print(f"Caching {len(unique_task_ids)} OpenML task(s) into OpenML cache: {cache_dir}")

        for task_id in tqdm(unique_task_ids, desc="Caching OpenML tasks"):
            openml.tasks.get_task(
                task_id,
                download_data=True,
                download_qualities=True,
                download_splits=True,
            )
