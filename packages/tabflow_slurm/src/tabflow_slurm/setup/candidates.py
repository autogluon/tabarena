"""The scheduler-side projection of core ``Job``s + the Ray-side cache filter.

Enumeration, constraint filtering, and the cache layout all live in tabarena core now
(:func:`~tabarena.benchmark.experiment.job.build_jobs`,
:func:`~tabarena.benchmark.experiment.job.filter_jobs_by_constraints`,
:func:`~tabarena.benchmark.experiment.experiment_runner_api.job_cache_exists`). What
remains here is the thin, pickle-friendly projection a SLURM setup needs per job —
its identity coordinates plus the dataset shape the scheduler's bundling rules read —
and the batched cache check fanned out across Ray workers.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from tabarena.benchmark.experiment import job_cache_exists

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import Job
    from tabarena.benchmark.task.metadata import SplitMetadata, TabArenaTaskMetadata


@dataclass(frozen=True)
class JobCandidate:
    """One core ``Job``, projected for scheduling.

    Carries the job's identity coordinates (``experiment_name``, ``dataset``,
    ``fold``, ``repeat`` — exactly what :meth:`to_item` serializes into the array-task
    JSON), the ``task_id_str`` the cache check keys on, and the dataset shape the
    scheduler's bundling rules read (`bundle_size_per_dataset`, large-dataset
    singletons). The live ``Experiment`` deliberately stays behind in the ``Job`` /
    ``JobBatch`` — candidates are pickled to Ray workers and must stay light.
    """

    experiment_name: str
    dataset: str
    fold: int
    repeat: int
    task_id_str: str
    n_features: int
    n_samples_train_per_fold: int

    @classmethod
    def from_job(cls, job: Job, *, task: TabArenaTaskMetadata, split: SplitMetadata) -> JobCandidate:
        """Project a core ``Job`` given its task + split metadata from the collection."""
        return cls(
            experiment_name=job.experiment.name,
            dataset=job.task.dataset,
            fold=job.task.fold,
            repeat=job.task.repeat,
            task_id_str=task.task_id_str,
            n_features=split.num_features_train,
            n_samples_train_per_fold=split.num_instances_train,
        )

    def to_item(self) -> dict:
        """The per-array-task JSON item: the job's self-describing coordinates.

        The runner resolves ``experiment`` by name against the shipped ``JobBatch``
        artifact and ``dataset`` against its collection, so the item carries no
        positional index into any file.
        """
        return {
            "experiment": self.experiment_name,
            "dataset": self.dataset,
            "fold": self.fold,
            "repeat": self.repeat,
        }


def is_job_cached_batch(*, candidates: list[JobCandidate], output_dir: str) -> list[bool]:
    """Whether each candidate's results cache already exists (batched for Ray workers).

    Module-level so Ray can pickle it. Delegates to the core, writer-aligned
    :func:`job_cache_exists`, so this pre-check can never drift from the cache layout
    the run engine writes.
    """
    return [
        job_cache_exists(
            output_dir=output_dir,
            method_name=c.experiment_name,
            task_id_str=c.task_id_str,
            fold=c.fold,
            repeat=c.repeat,
        )
        for c in candidates
    ]
