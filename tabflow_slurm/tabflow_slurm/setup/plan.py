"""Per-model overrides on top of a default benchmark setup.

`TabArenaBenchmarkSetup` describes a single homogeneous run (one resources /
scheduler / tasks / experiment combination). Real launches need different
hardware, partitions, or data-size bands per model. This module adds a thin
orchestration layer on top: define a base/default setup plus a list of
`ModelJob`s, each with optional dict overrides, and `TabArenaBenchmarkPlan`
expands them into the matching `TabArenaBenchmarkSetup`s and aggregates their
run commands.

Jobs whose effective `(resources, scheduler, tasks, experiment-minus-models)`
are identical are auto-merged into one run (one configs YAML + one set of array
tasks); `name` only labels the `parallel_benchmark_name`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from tabarena.benchmark.experiment import Experiment

from tabflow_slurm.setup.benchmark import TabArenaBenchmarkSetup

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import TabArenaExperimentBundle
    from tabarena.benchmark.task.metadata import TabArenaMetadataBundle

    from tabflow_slurm.setup.paths import PathSetup
    from tabflow_slurm.setup.resources import ResourcesSetup
    from tabflow_slurm.setup.scheduler import SchedulerSetup


def _apply_overrides(base: Any, overrides: dict[str, Any], label: str) -> Any:
    """Return `base` with `overrides` applied via `dataclasses.replace`.

    Empty `overrides` returns `base` unchanged (same object, so signature
    comparison can short-circuit on identity). Unknown keys raise `ValueError`
    listing the valid field names for `label` (e.g. "resources").
    """
    if not overrides:
        return base
    valid = {f.name for f in dataclasses.fields(base)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(
            f"Unknown {label} override key(s) {sorted(unknown)}. "
            f"Valid {label} keys: {sorted(valid)}."
        )
    return replace(base, **overrides)


@dataclass
class SingleModel:
    """One model entry: a name plus the number of configs to run for it.

    The typed counterpart of the `(name, n_configs)` tuples that
    `TabArenaExperimentBundle.models` accepts. Construct directly or via
    `SingleModel.from_input` (which also accepts a bare name or a tuple).
    """

    name: str
    n_configs: int | str | dict = 0
    """Random hyperparameter configs to run for this model:
        - int: that many random configs (0 = only the default config).
        - "all": `n_random_configs`-many configs (resolved by the bundle).
        - dict: kwargs for an AGExperiment (AutoGluon full-pipeline models).
    """

    @classmethod
    def from_input(cls, model: SingleModel | tuple | str) -> SingleModel:
        """Normalize a model spec into a `SingleModel`.

        Accepts a `SingleModel` (returned as-is), a `(name, n_configs)` tuple,
        or a bare model name string (uses the default `n_configs`).
        """
        if isinstance(model, SingleModel):
            return model
        if isinstance(model, str):
            return cls(name=model)
        if isinstance(model, tuple):
            return cls(*model)
        raise TypeError(
            f"Cannot interpret {model!r} as a model. Expected a SingleModel, "
            f"a (name, n_configs) tuple, or a model name string."
        )

    def to_entry(self) -> tuple[str, int | str | dict]:
        """The `(name, n_configs)` tuple consumed by `TabArenaExperimentBundle`."""
        return (self.name, self.n_configs)


# A model spec accepted by `ModelJob.models`. Pre-built `Experiment` objects
# pass through untouched; everything else is normalized to `SingleModel`.
ModelEntry = SingleModel | tuple | str | Experiment


@dataclass
class ModelJob:
    """One or more models sharing a set of overrides on the plan's base setup.

    `models` is normalized in `__post_init__` to a list of `SingleModel`
    (tuples/strings resolved via `SingleModel.from_input`), with pre-built
    `Experiment` objects passed through unchanged. The four override dicts are
    applied onto the plan's corresponding base setup via `dataclasses.replace`
    (see `_apply_overrides`); empty dicts leave the base untouched.
    """

    models: ModelEntry | list[ModelEntry]
    """One model, or a list of models, that share this job's overrides."""
    name: str | None = None
    """`parallel_benchmark_name` for this job's group. Jobs that merge into one
    group must agree on a non-None name (else `ValueError`); a fully unnamed
    group auto-derives `group{idx}`."""
    resources: dict[str, Any] = field(default_factory=dict)
    """Field overrides applied to the plan's base `ResourcesSetup`
    (e.g. `num_gpus`, `time_limit`, `memory_limit`, `fake_memory_for_estimates`)."""
    scheduler: dict[str, Any] = field(default_factory=dict)
    """Field overrides applied to the plan's base `SchedulerSetup`
    (e.g. `gpu_partition`, `bundle_size`, `array_job_limit`)."""
    tasks: dict[str, Any] = field(default_factory=dict)
    """Field overrides applied to the plan's base `TabArenaMetadataBundle`
    (e.g. `n_train_samples_to_run`, `dataset_names_to_run`)."""
    experiment: dict[str, Any] = field(default_factory=dict)
    """Field overrides applied to the plan's base `TabArenaExperimentBundle`
    (e.g. `model_agnostic_preprocessing`). The `models` key is forbidden here —
    use `models` above."""
    ignore_cache: bool = False
    """If True, overwrite the cache and rerun all of this job's items. Part of the
    grouping signature: jobs differing only in `ignore_cache` are not merged."""

    def __post_init__(self) -> None:
        models = self.models if isinstance(self.models, list) else [self.models]
        self.models = [m if isinstance(m, Experiment) else SingleModel.from_input(m) for m in models]

    def _model_entries(self) -> list:
        """The bundle-ready `models` list: `SingleModel` -> tuple, `Experiment` -> itself."""
        return [m if isinstance(m, Experiment) else m.to_entry() for m in self.models]


@dataclass
class TabArenaBenchmarkPlan:
    """A base benchmark setup plus per-model jobs that override parts of it.

    `build_setups` expands the `model_jobs` into `TabArenaBenchmarkSetup`s
    (merging jobs with identical effective settings); `setup_jobs` runs them all
    and returns the aggregated run commands.
    """

    benchmark_name: str
    """Unique name of the benchmark; shared by every generated run (the
    per-group `parallel_benchmark_name` keeps their setup artifacts distinct)."""
    model_jobs: list[ModelJob]
    """The models to run, each with optional per-model overrides."""

    # Base / default building blocks (the same objects TabArenaBenchmarkSetup takes).
    tasks_to_run_setup: TabArenaMetadataBundle
    """Default tasks. Per-job `tasks` overrides are applied on top."""
    experiment_bundle: TabArenaExperimentBundle
    """Default experiment settings. Its `models` field is ignored (a template);
    each group's models come from its `ModelJob`s. Per-job `experiment`
    overrides are applied on top."""
    path_setup: PathSetup
    """Paths for the benchmark (shared across all generated runs)."""
    scheduler_setup: SchedulerSetup
    """Default scheduler. Per-job `scheduler` overrides are applied on top."""
    resources_setup: ResourcesSetup
    """Default resources. Per-job `resources` overrides are applied on top."""

    def build_setups(self, num_ray_cpus: int | Literal["auto"] = "auto") -> list[TabArenaBenchmarkSetup]:
        """Expand the model jobs into one `TabArenaBenchmarkSetup` per group.

        Jobs are grouped by their effective `(resources, scheduler, tasks,
        experiment-minus-models, ignore_cache)` signature (compared by value);
        models from merged jobs are concatenated. See module docstring for the
        merge rule. `num_ray_cpus` is the CPU budget each generated setup uses
        when checking the cache / filtering jobs ("auto" = all available CPUs).
        """
        groups = self._group_jobs()
        setups: list[TabArenaBenchmarkSetup] = []
        for idx, group in enumerate(groups):
            group_name = group["name"] if group["name"] is not None else f"group{idx}"
            experiment_bundle = replace(group["experiment"], models=group["models"])
            setups.append(
                TabArenaBenchmarkSetup(
                    benchmark_name=self.benchmark_name,
                    parallel_safe_benchmark_name=f"{self.benchmark_name}_{group_name}",
                    tasks_to_run_setup=group["tasks"],
                    experiment_bundle=experiment_bundle,
                    path_setup=self.path_setup,
                    scheduler_setup=group["scheduler"],
                    resources_setup=group["resources"],
                    ignore_cache=group["ignore_cache"],
                    num_ray_cpus=num_ray_cpus,
                )
            )
        return setups

    def _group_jobs(self) -> list[dict]:
        """Resolve each job's overrides and merge jobs with equal signatures.

        Returns a list of group dicts (in first-appearance order) holding the
        effective `resources`/`scheduler`/`tasks`/`experiment` (with empty
        `models`), the concatenated bundle-ready `models`, and the reconciled
        `name`.
        """
        groups: list[dict] = []
        for job in self.model_jobs:
            if "models" in job.experiment:
                raise ValueError(
                    "ModelJob.experiment must not set 'models'; use ModelJob.models instead."
                )
            resources = _apply_overrides(self.resources_setup, job.resources, "resources")
            scheduler = _apply_overrides(self.scheduler_setup, job.scheduler, "scheduler")
            tasks = _apply_overrides(self.tasks_to_run_setup, job.tasks, "tasks")
            # Zero out models so only non-model settings define the signature.
            experiment = _apply_overrides(self.experiment_bundle, {**job.experiment, "models": []}, "experiment")

            signature = (resources, scheduler, tasks, experiment, job.ignore_cache)
            match = next((g for g in groups if g["signature"] == signature), None)
            if match is None:
                groups.append(
                    {
                        "signature": signature,
                        "resources": resources,
                        "scheduler": scheduler,
                        "tasks": tasks,
                        "experiment": experiment,
                        "ignore_cache": job.ignore_cache,
                        "models": list(job._model_entries()),
                        "name": job.name,
                    }
                )
            else:
                match["models"].extend(job._model_entries())
                match["name"] = self._reconcile_name(match["name"], job.name)
        return groups

    @staticmethod
    def _reconcile_name(current: str | None, new: str | None) -> str | None:
        """Combine the names of two jobs merging into the same group."""
        if new is None:
            return current
        if current is not None and current != new:
            raise ValueError(
                f"Jobs with identical settings merge into one run but have conflicting "
                f"names {current!r} and {new!r}. Use the same name (or leave one unset)."
            )
        return new

    def setup_jobs(self, num_ray_cpus: int | Literal["auto"] = "auto") -> list[str]:
        """Generate the job files for every group and return all run commands.

        Calls `TabArenaBenchmarkSetup.setup_jobs` on each generated setup,
        dropping groups with no work to launch, and returns the flat list of
        run commands. `num_ray_cpus` is forwarded to `build_setups`.
        """
        setups = self.build_setups(num_ray_cpus=num_ray_cpus)
        all_commands: list[str] = []
        for setup in setups:
            commands = setup.setup_jobs()
            if commands:
                all_commands.extend(commands)

        print(
            f"##### Plan {self.benchmark_name}: {len(setups)} run(s), "
            f"{len(all_commands)} command(s) to launch."
            + ("\n" + "\n".join(all_commands) + "\n" if all_commands else " (nothing to run)")
        )
        return all_commands
