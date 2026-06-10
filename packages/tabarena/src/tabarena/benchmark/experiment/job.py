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
    from tabarena.benchmark.experiment.bundle import ModelConstraints
    from tabarena.benchmark.experiment.experiment_constructor import Experiment
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.schema import SplitMetadata, TabArenaTaskMetadata


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

    # --- (De)serialization -----------------------------------------------------------
    def to_dict(self) -> dict:
        """Serialize this single job to a fully self-contained dict.

        The experiment is *inlined* (its ``to_yaml_dict`` form), so the dict can be
        stored / shipped / re-run with no other file — the atomic portable unit, e.g.
        for re-running one failed work unit. For a whole sweep, prefer
        :class:`~tabarena.benchmark.experiment.job_batch.JobBatch`, which stores each
        experiment once instead of inlining it into every job.
        """
        return {
            "experiment": self.experiment.to_yaml_dict(),
            "dataset": self.task.dataset,
            "fold": self.task.fold,
            "repeat": self.task.repeat,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Job:
        """Reconstruct a job from its :meth:`to_dict` form (experiment inlined)."""
        from tabarena.benchmark.experiment.experiment_constructor import YamlSingleExperimentSerializer

        return cls(
            experiment=YamlSingleExperimentSerializer.parse_method(data["experiment"]),
            task=Task(dataset=data["dataset"], fold=data["fold"], repeat=data["repeat"]),
        )


def build_jobs(
    experiments: list[Experiment],
    task_metadata: TaskMetadataCollection,
) -> list[Job]:
    """Expand ``experiments`` x the collection's splits into a flat ``list[Job]``.

    The one grid enumerator shared by every front end: each experiment is paired with
    every ``(dataset, fold, repeat)`` split of ``task_metadata`` (a sparse /
    non-rectangular collection is respected — exactly its splits are used, mirroring
    :meth:`ExperimentBatchRunner.run_all`). Jobs are ordered task -> split ->
    experiment, matching the runner's execution grouping.
    """
    return [
        Job.create(experiment, dataset, fold=fold, repeat=repeat)
        for (dataset, fold, repeat) in task_metadata.dataset_fold_repeats()
        for experiment in experiments
    ]


def _experiment_ag_key(experiment: Experiment) -> str:
    """Resolve the AutoGluon model key a live experiment is constrained under.

    The live-object counterpart of the serialized-config resolution (see
    ``tabflow_slurm.setup.candidates._resolve_ag_key``): the wrapped class is read from
    the experiment's captured constructor args (``model_cls`` for model experiments,
    ``method_cls`` for plain ones — the same keys the YAML form carries) and its
    ``ag_key`` is used. Full-pipeline AutoGluon experiments (neither key) map to
    ``"AutoGluon"``; classes without an ``ag_key`` fall back to their class name, which
    simply won't match any constraint key (i.e. unconstrained).
    """
    ctor_args = getattr(experiment, "_locals", None) or {}
    cls_obj = ctor_args.get("model_cls") or ctor_args.get("method_cls")
    if cls_obj is not None:
        return getattr(cls_obj, "ag_key", None) or cls_obj.__name__
    if experiment.name.startswith("AutoGluon"):
        return "AutoGluon"
    return experiment.name


def _split_index(
    task_metadata: TaskMetadataCollection,
) -> dict[tuple[str, int, int], tuple[TabArenaTaskMetadata, SplitMetadata]]:
    """Index the collection's splits by ``(dataset, fold, repeat)`` for shape lookups."""
    index: dict[tuple[str, int, int], tuple[TabArenaTaskMetadata, SplitMetadata]] = {}
    for ttm in task_metadata:
        for split in ttm.splits_metadata.values():
            index.setdefault((ttm.tabarena_task_name, split.fold, split.repeat), (ttm, split))
    return index


def filter_jobs_by_constraints(
    jobs: list[Job],
    *,
    model_constraints: dict[str, ModelConstraints],
    task_metadata: TaskMetadataCollection,
    verbose: bool = True,
) -> list[Job]:
    """Drop jobs whose dataset shape violates their model's :class:`ModelConstraints`.

    Each job's split shape (`n_features` / `n_classes` / train size / problem type) is
    looked up in ``task_metadata`` (raises if a job's split is not in the collection),
    and checked against ``model_constraints`` keyed by the experiment's AG model key.
    Experiments without a matching constraint entry always pass. Used by the SLURM
    dispatch filter and, opt-in, by :meth:`ExperimentBatchRunner.run_jobs`.
    """
    index = _split_index(task_metadata)
    kept: list[Job] = []
    for job in jobs:
        coords = job.task.as_triple()
        entry = index.get(coords)
        if entry is None:
            raise ValueError(
                f"Job split {coords} (experiment {job.experiment.name!r}) is not a split of `task_metadata`.",
            )
        ttm, split = entry
        constraints = model_constraints.get(_experiment_ag_key(job.experiment))
        if constraints is None or constraints.applies(
            n_features=split.num_features_train,
            n_classes=split.num_classes_train,
            n_samples_train_per_fold=split.num_instances_train,
            problem_type=ttm.problem_type,
        ):
            kept.append(job)
    if verbose and len(kept) != len(jobs):
        print(f"Model constraints filtered {len(jobs) - len(kept)}/{len(jobs)} job(s); {len(kept)} remain.")
    return kept
