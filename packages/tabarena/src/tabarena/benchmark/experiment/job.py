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
spec-keyed ``_JobSpec``, then dispatches the whole list through the shared engine
(``_run_job_specs`` in ``experiment_runner_api``) in a single task-grouped pass — which
expands the specs into the runner's fully-resolved ``_Job`` work units and loads each task
once, even when several experiments share it.

The module also hosts the helpers that operate on jobs: :func:`build_jobs` (the one grid
enumerator), :func:`filter_jobs_by_constraints` (dataset-shape compatibility), and
:class:`JobBatch` — a complete sweep (jobs + the task metadata they resolve against) as
one self-contained on-disk artifact. A single :class:`Job` serializes standalone with its
experiment inlined (:meth:`Job.to_dict`); a batch stores each experiment once, by name.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tabarena.benchmark.experiment.experiment_constructor import Experiment
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
    from tabarena.benchmark.task.metadata.schema import SplitMetadata, TabArenaTaskMetadata
    from tabarena.caching import CacheConfig


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
        :class:`JobBatch`, which stores each experiment once instead of inlining it
        into every job.
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

    Pairs that violate an experiment's attached
    :class:`~tabarena.benchmark.experiment.model_constraints.ModelConstraints` are
    dropped during enumeration (see :func:`filter_jobs_by_constraints`); experiments
    without constraints run on every split.
    """
    jobs = [
        Job.create(experiment, dataset, fold=fold, repeat=repeat)
        for (dataset, fold, repeat) in task_metadata.dataset_fold_repeats()
        for experiment in experiments
    ]
    return filter_jobs_by_constraints(jobs, task_metadata=task_metadata)


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
    task_metadata: TaskMetadataCollection,
    verbose: bool = True,
) -> list[Job]:
    """Drop jobs whose dataset shape violates their experiment's attached constraints.

    Constraints are a property of the experiment
    (:attr:`Experiment.model_constraints
    <tabarena.benchmark.experiment.experiment_constructor.Experiment>`, attached at
    build time by ``TabArenaExperimentBundle`` or explicitly by the user). For each
    *constrained* job, its split shape (`n_features` / `n_classes` / train size /
    problem type) is looked up in ``task_metadata`` (raises if that job's split is not
    in the collection) and checked via :meth:`ModelConstraints.applies`. Jobs whose
    experiment carries no constraints always pass — without any collection lookup, so
    unconstrained sweeps are unaffected by this filter.

    Applied automatically by :func:`build_jobs` and
    :meth:`ExperimentBatchRunner.run_jobs`.
    """
    index: dict[tuple[str, int, int], tuple[TabArenaTaskMetadata, SplitMetadata]] | None = None
    kept: list[Job] = []
    for job in jobs:
        constraints = getattr(job.experiment, "model_constraints", None)
        if constraints is None:
            kept.append(job)
            continue
        if index is None:
            index = _split_index(task_metadata)
        coords = job.task.as_triple()
        entry = index.get(coords)
        if entry is None:
            raise ValueError(
                f"Job split {coords} (experiment {job.experiment.name!r}) is not a split of `task_metadata`, "
                f"so its model constraints cannot be checked.",
            )
        ttm, split = entry
        if constraints.applies(
            n_features=split.num_features_train,
            n_classes=split.num_classes_train,
            n_samples_train_per_fold=split.num_instances_train,
            problem_type=ttm.problem_type,
        ):
            kept.append(job)
    if verbose and len(kept) != len(jobs):
        print(f"Model constraints filtered {len(jobs) - len(kept)}/{len(jobs)} job(s); {len(kept)} remain.")
    return kept


_EXPERIMENTS_FILE = "experiments.yaml"
_TASK_METADATA_FILE = "task_metadata.csv"
_JOBS_FILE = "jobs.json"
_CACHE_CONFIG_FILE = "cache_config.json"


@dataclass
class JobBatch:
    """A complete, self-contained benchmark sweep: a ``list[Job]`` plus the task
    metadata its datasets resolve against.

    ``save``/``load`` round-trip the batch through a directory artifact with each
    piece stored once in its native on-disk format:

    * ``experiments.yaml`` — every distinct experiment, stored *once* and keyed by its
      unique ``name`` (the results-cache identity), in the standard
      :class:`~tabarena.benchmark.experiment.experiment_constructor.YamlExperimentSerializer`
      format.
    * ``task_metadata.csv`` — the collection in its native one-row-per-split schema
      (the same format as the committed reference CSVs), so the loading side resolves
      dataset names / task ids / shapes identically to the authoring side.
    * ``jobs.json`` — the (tiny) job list: ``(experiment name, dataset, fold, repeat)``
      coordinates only.
    * ``cache_config.json`` — the optional :class:`~tabarena.caching.CacheConfig` (only written
      when set), so the compute node configures the same OpenML / HuggingFace / TabArena cache
      locations the run was set up with, with no out-of-band wiring.

    Loading a saved directory needs nothing else — ``JobBatch.load(path)`` reconstructs
    the ``list[Job]`` plus the collection (and the ``cache_config`` if present), ready for
    :meth:`ExperimentBatchRunner.run_jobs`. This is the artifact a head node ships to
    compute nodes (each array task runs a slice of ``jobs``), and the user-facing way to
    serialize a sweep and re-run it later. Construction validates the batch is
    internally consistent (see ``__post_init__``).
    """

    jobs: list[Job]
    task_metadata: TaskMetadataCollection
    cache_config: CacheConfig | None = None
    """Optional cache locations (OpenML / HuggingFace / TabArena) the run was configured with.
    Persisted to ``cache_config.json`` and applied by the runner on the compute node."""

    def __post_init__(self) -> None:
        """Validate name-uniqueness and that every job's split exists in the collection.

        ``experiment.name`` is the results-cache key and the on-disk reference key, so
        two *different* experiments sharing a name would cross-contaminate caches and
        collide in ``experiments.yaml``. Identical experiments may repeat across jobs
        (that is the normal sweep shape).
        """
        by_name: dict[str, Experiment] = {}
        for job in self.jobs:
            experiment = job.experiment
            existing = by_name.setdefault(experiment.name, experiment)
            if existing is not experiment and existing.to_yaml_str() != experiment.to_yaml_str():
                raise ValueError(
                    f"Two different experiments share the name {experiment.name!r}; names must be "
                    f"unique because they are the cache / serialization identity.",
                )

        valid_splits = set(self.task_metadata.dataset_fold_repeats())
        invalid = [job.task.as_triple() for job in self.jobs if job.task.as_triple() not in valid_splits]
        if invalid:
            invalid_str = "\n\t".join(str(t) for t in invalid[:20])
            raise ValueError(
                f"{len(invalid)} job(s) reference (dataset, fold, repeat) splits not in `task_metadata`:"
                f"\n\t{invalid_str}",
            )

    @property
    def experiments(self) -> list[Experiment]:
        """The distinct experiments of this batch, in first-seen job order."""
        by_name: dict[str, Experiment] = {}
        for job in self.jobs:
            by_name.setdefault(job.experiment.name, job.experiment)
        return list(by_name.values())

    # ------------------------------------------------------------------ persistence
    def save(self, path: str | Path) -> Path:
        """Write the batch to directory ``path`` (created if missing); returns the path."""
        from tabarena.benchmark.experiment.experiment_constructor import YamlExperimentSerializer

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        YamlExperimentSerializer.to_yaml(experiments=self.experiments, path=str(path / _EXPERIMENTS_FILE))
        self.task_metadata.to_dataframe().to_csv(path / _TASK_METADATA_FILE, index=False)
        job_records = [
            {
                "experiment": job.experiment.name,
                "dataset": job.task.dataset,
                "fold": job.task.fold,
                "repeat": job.task.repeat,
            }
            for job in self.jobs
        ]
        with (path / _JOBS_FILE).open("w") as f:
            json.dump({"jobs": job_records}, f)
        if self.cache_config is not None:
            with (path / _CACHE_CONFIG_FILE).open("w") as f:
                json.dump(self.cache_config.to_dict(), f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> JobBatch:
        """Reconstruct a batch from a directory written by :meth:`save`."""
        from tabarena.benchmark.experiment.experiment_constructor import YamlExperimentSerializer
        from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection

        path = Path(path)
        experiments = YamlExperimentSerializer.from_yaml(path=str(path / _EXPERIMENTS_FILE))
        experiment_by_name = {experiment.name: experiment for experiment in experiments}
        task_metadata = TaskMetadataCollection.from_source(path / _TASK_METADATA_FILE)

        cache_config = None
        cache_config_path = path / _CACHE_CONFIG_FILE
        if cache_config_path.exists():
            from tabarena.caching import CacheConfig

            with cache_config_path.open() as f:
                cache_config = CacheConfig.from_dict(json.load(f))

        with (path / _JOBS_FILE).open() as f:
            job_records = json.load(f)["jobs"]

        jobs: list[Job] = []
        for record in job_records:
            experiment = experiment_by_name.get(record["experiment"])
            if experiment is None:
                raise ValueError(
                    f"Job references experiment {record['experiment']!r}, which is not in "
                    f"{_EXPERIMENTS_FILE} (has: {sorted(experiment_by_name)}).",
                )
            jobs.append(
                Job.create(experiment, record["dataset"], fold=record["fold"], repeat=record["repeat"]),
            )
        return cls(jobs=jobs, task_metadata=task_metadata, cache_config=cache_config)
