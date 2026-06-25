"""Per-model overrides on top of a default benchmark setup.

`TabArenaBenchmarkSetup` describes a single homogeneous run (one resources /
scheduler / tasks / experiment combination). Real launches need different
hardware, partitions, or data-size bands per model. This module adds a thin
orchestration layer on top: define a base/default setup plus a list of
`ModelJob`s, each with optional dict overrides, and `TabArenaBenchmarkPlan`
expands them into the matching `TabArenaBenchmarkSetup`s and aggregates their
run commands.

Jobs whose effective `(resources, scheduler, task_subset, experiment-minus-models)`
are identical are auto-merged into one run (one configs YAML + one set of array
tasks); `name` only labels the `parallel_benchmark_name`.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Any, Literal

from tabarena.benchmark.experiment import BeyondArenaExperimentBundle, Experiment
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import BeyondArenaContext
from tabflow_slurm.setup.benchmark import TabArenaBenchmarkSetup
from tabflow_slurm.setup.resources import BeyondArenaResourcesSetup
from tabflow_slurm.setup.scheduler import GCPSlurmSetup

if TYPE_CHECKING:
    from tabarena.benchmark.experiment import TabArenaExperimentBundle
    from tabarena.contexts import AbstractArenaContext
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
            f"Unknown {label} override key(s) {sorted(unknown)}. Valid {label} keys: {sorted(valid)}.",
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
            f"a (name, n_configs) tuple, or a model name string.",
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
    `Experiment` objects passed through unchanged. The `resources`/`scheduler`/
    `experiment` override dicts are applied onto the plan's corresponding base setup
    via `dataclasses.replace` (see `_apply_overrides`); empty dicts leave the base
    untouched. `tasks` is normalized to a typed `TaskSubset` and merged onto the
    plan's `task_subset` (see `_group_jobs`).
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
    tasks: TaskSubset | dict[str, Any] | None = None
    """Per-job scoping merged on top of the plan's `task_subset` and forwarded to
    `context.build_jobs` (i.e. `TaskMetadataCollection.subset_tasks`). A `TaskSubset`
    (or a dict that resolves to one) — e.g. `TaskSubset(n_train_samples=(0, 10_000))`.
    Fields set here override the plan-level `task_subset` for the same field (the job
    wins); fields left unset fall back to the plan. Normalized to a `TaskSubset` in
    `__post_init__`. Jobs differing in `tasks` are not merged into one group."""
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
        self.tasks = TaskSubset.from_input(self.tasks)

    def _model_entries(self) -> list:
        """The bundle-ready `models` list: `SingleModel` -> tuple, `Experiment` -> itself."""
        return [m if isinstance(m, Experiment) else m.to_entry() for m in self.models]


_SUMMARY_BAR = "=" * 78


def _format_models(models: list) -> str:
    """Comma-separated model names for a run (tuples -> name, Experiment -> .name)."""
    names = [m[0] if isinstance(m, tuple) else getattr(m, "name", str(m)) for m in models]
    return ", ".join(str(n) for n in names) if names else "(none)"


def _format_resources(resources: ResourcesSetup) -> str:
    """One-line compute summary: GPUs / CPUs / memory / per-config time budget."""
    cpus = resources.num_cpus if resources.num_cpus is not None else "all"
    mem = f"{resources.memory_limit}GB" if resources.memory_limit is not None else "all RAM"
    return f"{resources.num_gpus} GPU, {cpus} CPU, {mem}, ~{resources.time_limit / 3600:g}h/config"


def _format_partition(scheduler: SchedulerSetup, resources: ResourcesSetup) -> str:
    """The partition this run lands on (GPU vs CPU), or the scheduler type if it has none."""
    attr = "gpu_partition" if resources.num_gpus > 0 else "cpu_partition"
    return getattr(scheduler, attr, None) or type(scheduler).__name__


@dataclass(kw_only=True)
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
    context: AbstractArenaContext
    """The arena context that owns the task-metadata collection + subset predicates
    (e.g. `TabArenaContext()` / `BeyondArenaContext()`). Jobs are enumerated through
    `context.build_jobs`. Scope it with `build_kwargs` below rather than pre-filtering
    the collection, so the context's named subset predicates stay available."""
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
    task_subset: TaskSubset | dict[str, Any] | None = None
    """Plan-level scoping forwarded (via `as_kwargs()`) to `context.build_jobs`, i.e.
    `TaskMetadataCollection.subset_tasks`. A `TaskSubset` (or a dict that resolves to one) —
    e.g. `TaskSubset(subset="lite", dataset_names=[...])`. `None`/empty runs the context's full
    collection. Each `ModelJob.tasks` is merged on top per group (the job wins per field)."""
    prefetch_model_weights: bool = True
    """If True, `setup_jobs` warms the weights of any selected foundation models on this (head)
    node before emitting jobs, so parallel/offline compute nodes find them cached. Set False to
    skip (e.g. weights already present, or no network on the head node)."""

    def build_setups(self, num_ray_cpus: int | Literal["auto"] = "auto") -> list[TabArenaBenchmarkSetup]:
        """Expand the model jobs into one `TabArenaBenchmarkSetup` per group.

        Jobs are grouped by their effective `(resources, scheduler, task_subset,
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
                    context=self.context,
                    task_subset=group["task_subset"],
                    experiment_bundle=experiment_bundle,
                    path_setup=self.path_setup,
                    scheduler_setup=group["scheduler"],
                    resources_setup=group["resources"],
                    ignore_cache=group["ignore_cache"],
                    num_ray_cpus=num_ray_cpus,
                ),
            )
        return setups

    def _group_jobs(self) -> list[dict]:
        """Resolve each job's overrides and merge jobs with equal signatures.

        Returns a list of group dicts (in first-appearance order) holding the
        effective `resources`/`scheduler`/`task_subset`/`experiment` (with empty
        `models`), the concatenated bundle-ready `models`, and the reconciled
        `name`.
        """
        groups: list[dict] = []
        for job in self.model_jobs:
            if "models" in job.experiment:
                raise ValueError(
                    "ModelJob.experiment must not set 'models'; use ModelJob.models instead.",
                )
            resources = _apply_overrides(self.resources_setup, job.resources, "resources")
            scheduler = _apply_overrides(self.scheduler_setup, job.scheduler, "scheduler")
            # Per-job `tasks` scoping (a TaskSubset) is layered on top of the plan-level
            # `task_subset` (job fields win) and forwarded to `context.build_jobs`. Equal
            # merged TaskSubsets compare equal, so jobs with the same scoping share one group.
            task_subset = TaskSubset.from_input(self.task_subset).merged_with(job.tasks)
            # Zero out models so only non-model settings define the signature.
            experiment = _apply_overrides(self.experiment_bundle, {**job.experiment, "models": []}, "experiment")

            signature = (resources, scheduler, task_subset, experiment, job.ignore_cache)
            match = next((g for g in groups if g["signature"] == signature), None)
            if match is None:
                groups.append(
                    {
                        "signature": signature,
                        "resources": resources,
                        "scheduler": scheduler,
                        "task_subset": task_subset,
                        "experiment": experiment,
                        "ignore_cache": job.ignore_cache,
                        "models": list(job._model_entries()),
                        "name": job.name,
                    },
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
                f"names {current!r} and {new!r}. Use the same name (or leave one unset).",
            )
        return new

    def setup_jobs(self, num_ray_cpus: int | Literal["auto"] = "auto") -> list[str]:
        """Generate the job files for every run and return all run commands.

        Prepares each run (`TabArenaBenchmarkSetup`) in turn — printing a banner
        per run so the interleaved cache/filter logs are attributable — then
        prints one consolidated summary and the final list of commands to launch.
        `num_ray_cpus` is forwarded to `build_setups`.
        """
        setups = self.build_setups(num_ray_cpus=num_ray_cpus)
        n = len(setups)

        if self.prefetch_model_weights:
            self._prefetch_model_weights()

        print(f"\n{_SUMMARY_BAR}\nBenchmark plan '{self.benchmark_name}': preparing {n} run(s)\n{_SUMMARY_BAR}")

        runs: list[tuple[str, str, list[str]]] = []
        for idx, setup in enumerate(setups, start=1):
            label = setup._safe_benchmark_name
            models = _format_models(setup.experiment_bundle.models)
            print(
                f"\n----- [{idx}/{n}] run '{label}' -----"
                f"\n  models:    {models}"
                f"\n  resources: {_format_resources(setup.resources_setup)}"
                f"\n  partition: {_format_partition(setup.scheduler_setup, setup.resources_setup)}",
            )
            commands = setup.setup_jobs(print_run_commands=False) or []
            runs.append((label, models, commands))

        return self._print_summary_and_collect(runs)

    def selected_model_names(self) -> list[str]:
        """Unique benchmark model names across all jobs (first-appearance order)."""
        names: list[str] = []
        for job in self.model_jobs:
            for entry in job._model_entries():
                name = entry[0] if isinstance(entry, tuple) else getattr(entry, "name", None)
                if name is not None and name not in names:
                    names.append(name)
        return names

    def _prefetch_model_weights(self) -> None:
        """Warm the weights of any selected foundation models on this node before dispatch."""
        from tabarena.models.prefetch import prefetch_weights

        model_names = self.selected_model_names()
        print(
            f"\n{_SUMMARY_BAR}"
            f"\nPrefetching foundation-model weights for: {', '.join(model_names) or '(none)'}"
            f"\n{_SUMMARY_BAR}",
        )
        prefetch_weights(model_names)

    def _print_summary_and_collect(self, runs: list[tuple[str, str, list[str]]]) -> list[str]:
        """Print the consolidated per-run summary + final command list; return all commands."""
        all_commands = [cmd for _, _, commands in runs for cmd in commands]

        print(
            f"\n{_SUMMARY_BAR}"
            f"\nPlan '{self.benchmark_name}' summary "
            f"— {len(runs)} run(s), {len(all_commands)} command(s) to launch"
            f"\n{_SUMMARY_BAR}",
        )
        prefix = f"{self.benchmark_name}_"
        for label, models, commands in runs:
            short = label[len(prefix) :] if label.startswith(prefix) else label
            status = f"{len(commands)} command(s)" if commands else "no jobs (all cached / filtered out)"
            print(f"  - {short:<10} {status:<36} [{models}]")

        if all_commands:
            print(f"\nRun the following {len(all_commands)} command(s) to launch the jobs:\n")
            print("\n\n".join(all_commands) + "\n")
        else:
            print("\nNothing to launch — all runs are already cached / filtered out.\n")
        return all_commands


@dataclass(kw_only=True)
class BeyondArenaBenchmarkPlan(TabArenaBenchmarkPlan):
    """:class:`TabArenaBenchmarkPlan` pre-wired with the BeyondArena building blocks.

    Every BeyondArena launch script pairs the same four pieces: the data-foundry
    :class:`~tabarena.contexts.beyondarena.context.BeyondArenaContext`, a
    :class:`~tabarena.benchmark.experiment.BeyondArenaExperimentBundle` template, the
    :class:`~tabflow_slurm.setup.scheduler.GCPSlurmSetup` scheduler, and
    :class:`~tabflow_slurm.setup.resources.BeyondArenaResourcesSetup`. This subclass
    supplies them as defaults (via ``default_factory`` so each plan gets its own
    instances), so a launch script only states what is actually being run —
    ``benchmark_name``, ``model_jobs``, ``path_setup``, and an optional ``task_subset``.

    Any default is still overridable by passing the field explicitly
    (e.g. ``scheduler_setup=GCPSlurmSetup(bundle_size=2)``). Mirrors the preset-subclass
    idiom already used for the building blocks themselves (``GCPSlurmSetup``,
    ``BeyondArenaResourcesSetup``). Construction builds a ``BeyondArenaContext`` only when
    ``context`` is omitted, which imports the optional ``data-foundry`` dependency.
    """

    context: AbstractArenaContext = field(default_factory=BeyondArenaContext)
    experiment_bundle: TabArenaExperimentBundle = field(default_factory=BeyondArenaExperimentBundle)
    scheduler_setup: SchedulerSetup = field(default_factory=GCPSlurmSetup)
    resources_setup: ResourcesSetup = field(default_factory=BeyondArenaResourcesSetup)
