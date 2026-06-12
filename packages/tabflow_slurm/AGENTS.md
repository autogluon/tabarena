# AGENTS.md — tabflow_slurm

Guidance for coding agents working in `packages/tabflow_slurm/`. Human-facing docs: [`README.md`](README.md).
Repo-wide guidance: [`../../AGENTS.md`](../../AGENTS.md).

## What this is

A package that generates and launches **SLURM** array jobs for TabArena benchmarks. The user
composes a `TabArenaBenchmarkPlan`, calls `setup_jobs()`, and gets `sbatch` command(s); each array
task fits one `(task, fold, repeat, config)` item at a time and caches the result. It is
self-contained and only depends on `tabarena`.

> It **is** a package (`pyproject.toml`, `tabflow_slurm/__init__.py`) and a uv-workspace member,
> installed editable from `packages/tabflow_slurm/`.

## Layout

```
tabflow_slurm/                      ← this folder (docs, examples, history, pyproject)
├── README.md, AGENTS.md, BENCHMARK_LOG.md
├── pyproject.toml                  ← declares the `tabflow_slurm` package; deps: tabarena
├── experiments/                    ← runnable setup/eval scripts (run_setup_*, run_eval_*)
└── tabflow_slurm/                  ← the package
    ├── __init__.py                 ← re-exports the public API
    ├── run_tabarena_experiment.py  ← runner: fits ONE item on a node (bundled script)
    ├── submit_template.sh          ← sbatch array script (parses job JSON via jq, calls runner)
    ├── slurm_utils.py              ← setup_slurm_job(): per-node OpenML cache + Ray init
    └── setup/                      ← the building blocks
        ├── plan.py                 ← TabArenaBenchmarkPlan (ENTRY POINT), ModelJob, SingleModel
        ├── benchmark.py            ← TabArenaBenchmarkSetup (INTERNAL per-run engine)
        ├── paths.py                ← PathSetup, get_run_script_path/get_submit_script_path
        ├── resources.py            ← ResourcesSetup + v0.1/BeyondArena presets
        └── scheduler.py            ← SchedulerSetup → SlurmSetup → GCPSlurmSetup (batching + sbatch)
```

## The public-API boundary

- **Entry point:** `TabArenaBenchmarkPlan` (in `setup/plan.py`). Everything users touch is
  re-exported from the top-level `tabflow_slurm` package and from `tabflow_slurm.setup`. When adding
  a public symbol, update **both** `__init__.py` `__all__` lists.
- **Internal:** `TabArenaBenchmarkSetup` (`setup/benchmark.py`) is the per-run engine. The plan
  builds/drives it (one per group). Don't push users toward it directly.

## Mental model of `setup_jobs()`

1. `plan.setup_jobs()` → `_prefetch_model_weights()` (head node) → `build_setups()`.
2. `build_setups()` → `_group_jobs()`: each `ModelJob`'s dict overrides are applied to the base
   building blocks via `dataclasses.replace` (its `tasks` dict instead *filters* the base
   `TaskMetadataCollection` via `subset_tasks`); jobs are **merged** when their
   `(resources, scheduler, tasks, experiment-with-models-zeroed, ignore_cache)` signature matches
   (collections compare by value). One `TabArenaBenchmarkSetup` per group.
3. Each setup's `get_jobs_to_run()`: ensure dirs → `tasks.materialize()` (downloads only the
   collection's tasks) → `experiment_bundle.build_experiments()` (attaches each experiment's
   `ModelConstraints`) → core `build_jobs` (experiments × splits; constraint-violating pairs are
   dropped during enumeration) → Ray cache check (core `job_cache_exists_batch` over plain
   coordinate tuples) → persist the surviving sweep as a `JobBatch` artifact
   (experiments.yaml + task_metadata.csv + jobs.json) → `scheduler.bundle_items()`.
4. `scheduler.get_run_commands()` writes the job JSON (splitting at `max_array_size`) and returns the
   `sbatch` command(s). The plan prints one consolidated summary.

Then: `sbatch … submit_template.sh <job.json>` → array task picks `jobs[SLURM_ARRAY_TASK_ID]` → runs
`run_tabarena_experiment.py` per item → `setup_slurm_job()` + `JobBatch.load()` +
`ExperimentBatchRunner.run_jobs()` (the exact same execution path as a local benchmark run).

## Conventions

- Building blocks are **frozen-ish dataclasses**; per-job customization is via override **dicts** on
  `ModelJob` (validated against the dataclass fields — unknown keys raise). Add new knobs as
  dataclass fields, not ad-hoc params.
- **Bundled scripts** (`run_tabarena_experiment.py`, `submit_template.sh`) live at the *package
  root* and are resolved via `get_run_script_path()` / `get_submit_script_path()` (relative to the
  install). `pyproject.toml` ships `*.sh` as package data — keep that if you add/rename shell files.
- The **job JSON** is the contract between Python and `submit_template.sh`: `{"defaults": {...},
  "jobs": [{"items": [{experiment, dataset, fold, repeat}]}]}`. If you change `defaults` keys, update
  the `jq` reads in `submit_template.sh` too. Items are self-describing coordinates: the runner
  resolves `experiment` by name and `dataset` against the shipped `JobBatch` artifact
  (`defaults.job_batch_dir`).
- `benchmark_name` vs `parallel_safe_benchmark_name`: the former is shared (output + log dirs); the
  latter (one per group, `<benchmark_name>_<group>`) namespaces the per-run `JobBatch` dir / job JSON
  so parallel runs don't clobber each other.

## Gotchas

- **The Ray cache check pickles plain `(method, task_id_str, fold, repeat)` tuples** to workers
  (never live experiments) and runs tabarena core's writer-aligned `job_cache_exists_batch` — don't
  re-derive the cache key/normalization in this package.
- **OpenML cache must be shared** between the head node (setup materializes tasks there) and workers
  (`--openml_cache_dir`). `PathSetup.ensure_runtime_dirs()` points the ambient cache at it during
  setup.
- **Ray on a shared filesystem:** `setup_slurm_job(setup_ray_for_slurm_shared_resources_environment=
  True)` gives each job a unique temp dir + plasma sizing; required when fold-fitting isn't
  sequential-local, else workers collide semi-randomly.
- **Time budget** for `--time` = `ResourcesSetup.time_limit_per_config × configs_per_job +
  time_limit_overhead` (hours). `configs_per_job` is the worst-case bundle size from
  `bundle_items()`.
- **`fake_memory_for_estimates`** intentionally lies to a model's memory estimator (VRAM-vs-RAM for
  TFMs); it does not change the SLURM `--mem`.

## Editing tasks

- Changing how jobs are launched/batched → `setup/scheduler.py` (add a `SchedulerSetup` subclass for
  a non-SLURM scheduler; the plan/engine are scheduler-agnostic).
- Changing what runs on a node → `run_tabarena_experiment.py` (Python) and/or `submit_template.sh`
  (bundle iteration); these two move together.
- New hardware preset → a `ResourcesSetup` subclass in `setup/resources.py`.
- The model **selection / config counts / preprocessing / constraints** come from `tabarena`'s
  `TabArenaExperimentBundle`, and the tasks from a `TaskMetadataCollection`
  (`from_preset` / `from_source` + `subset_tasks`), **not** here — this package only schedules.

## Tests & lint

Tests live in `../tst/tabflow_slurm/` (repo convention: `tst/`, mirroring the package path). Run
`ruff check .` / `ruff format .` from the repo root (config `../ruff.toml`; `from __future__ import
annotations` is required). After touching this package, also smoke-test a setup script's
`setup_jobs()` (it runs Ray + materialization locally).
