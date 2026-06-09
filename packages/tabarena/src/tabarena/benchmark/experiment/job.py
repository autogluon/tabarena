"""First-class ``(experiment, task-split)`` work units for the benchmark runner.

A :class:`Task` identifies a single task split — a dataset plus the ``(fold, repeat)``
coordinate of the split to evaluate. A :class:`Job` pairs a ``Task`` with the
:class:`~tabarena.benchmark.experiment.experiment_constructor.Experiment` to run on it,
so a job is *self-describing*: a ``list[Job]`` fully specifies a (possibly
non-rectangular) sweep without a separate ``methods`` x ``tasks`` cross product.

This promotes to a public type the unit the runner enumerates internally.
:meth:`ExperimentBatchRunner.run_jobs
<tabarena.benchmark.experiment.experiment_utils.ExperimentBatchRunner.run_jobs>` consumes a
``list[Job]`` directly: it resolves each ``Job``'s dataset name to a tid to build a
tid-keyed ``_JobSpec``, then dispatches the whole list through the shared engine
(``_run_job_specs`` in ``experiment_runner_api``) in a single task-grouped pass — which
expands the specs into the runner's fully-resolved ``_Job`` work units and loads each task
once, even when several experiments share it. The same unit is also projected to the SLURM
scheduler's ``JobCandidate`` (``tabflow_slurm.setup.candidates``).

Note this is the *in-memory* form (it holds a live ``Experiment``); the SLURM
``JobCandidate`` is the serialized projection that additionally carries the dataset shape
its Ray-side cache/constraint filter needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.benchmark.experiment.experiment_constructor import Experiment


@dataclass(frozen=True)
class Task:
    """A single task split: a dataset and the ``(fold, repeat)`` coordinate to evaluate.

    ``dataset`` is the dataset *name* as it appears in the runner's ``task_metadata`` —
    the same identifier accepted by :meth:`ExperimentBatchRunner.run`; the runner resolves
    it to an OpenML tid. ``repeat`` defaults to ``0`` (the results cache path is always
    ``{repeat}_{fold}``).

    This is intentionally the dataset-level identity plus a split coordinate (matching the
    runner's ``(dataset, fold, repeat)`` triples), distinct from the dataset-level
    :class:`~tabarena.benchmark.task.user_task.UserTask`. Carrying the shape/metadata here
    to make a job fully self-contained (no ``task_metadata`` lookup at run time) is a
    deliberate future extension, not part of this prototype.
    """

    dataset: str
    fold: int
    repeat: int = 0

    def __post_init__(self) -> None:
        if self.fold < 0:
            raise ValueError(f"fold must be >= 0, got {self.fold}.")
        if self.repeat < 0:
            raise ValueError(f"repeat must be >= 0, got {self.repeat}.")

    def as_triple(self) -> tuple[str, int, int]:
        """The ``(dataset, fold, repeat)`` triple the runner's "individual" path consumes."""
        return (self.dataset, self.fold, self.repeat)


@dataclass(frozen=True)
class Job:
    """One unit of benchmark work: run ``experiment`` on the ``task`` split.

    Frozen and self-describing — a ``list[Job]`` is a complete sweep specification, so
    different experiments may run on different tasks. De-duplication keys on
    ``(experiment.name, task)`` rather than on the ``Experiment`` value (see
    :meth:`ExperimentBatchRunner.run_jobs`); ``experiment.name`` is the cache identity and
    must be unique per distinct config.
    """

    experiment: Experiment
    task: Task

    @classmethod
    def create(cls, experiment: Experiment, dataset: str, fold: int, repeat: int = 0) -> Job:
        """Build a ``Job`` from an experiment and explicit ``(dataset, fold, repeat)`` coords."""
        return cls(experiment=experiment, task=Task(dataset=dataset, fold=fold, repeat=repeat))
