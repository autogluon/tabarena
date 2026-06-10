"""``JobBatch`` — a complete, self-contained benchmark sweep on disk.

A batch bundles everything needed to (re-)run a set of
:class:`~tabarena.benchmark.experiment.job.Job` work units, with each piece stored
once in its native on-disk format:

* ``experiments.yaml`` — every distinct experiment, stored *once* and keyed by its
  unique ``name`` (the results-cache identity), in the standard
  :class:`~tabarena.benchmark.experiment.experiment_constructor.YamlExperimentSerializer`
  format.
* ``task_metadata.csv`` — the :class:`~tabarena.benchmark.task.metadata.collection.
  TaskMetadataCollection` in its native one-row-per-split schema (the same format as
  the committed reference CSVs), so the loading side resolves dataset names / task ids
  / shapes identically to the authoring side.
* ``jobs.json`` — the (tiny) job list: ``(experiment name, dataset, fold, repeat)``
  coordinates only.

Loading a saved directory needs nothing else — ``JobBatch.load(path)`` reconstructs
the ``list[Job]`` plus the collection, ready for
:meth:`ExperimentBatchRunner.run_jobs`. This is the artifact a head node ships to
compute nodes (each array task runs a slice of ``jobs``), and the user-facing way to
serialize a sweep and re-run it later.

A *single* job serialized alone inlines its experiment instead (see
:meth:`Job.to_dict <tabarena.benchmark.experiment.job.Job.to_dict>`) — self-contained
at the job level where duplication doesn't matter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tabarena.benchmark.experiment.job import Job

if TYPE_CHECKING:
    from tabarena.benchmark.experiment.experiment_constructor import Experiment
    from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection

_EXPERIMENTS_FILE = "experiments.yaml"
_TASK_METADATA_FILE = "task_metadata.csv"
_JOBS_FILE = "jobs.json"


@dataclass
class JobBatch:
    """A ``list[Job]`` plus the task metadata its datasets resolve against.

    Construction validates the batch is internally consistent (see
    ``__post_init__``); ``save``/``load`` round-trip it through a directory artifact
    (see module docstring for the layout).
    """

    jobs: list[Job]
    task_metadata: TaskMetadataCollection

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
        return cls(jobs=jobs, task_metadata=task_metadata)
