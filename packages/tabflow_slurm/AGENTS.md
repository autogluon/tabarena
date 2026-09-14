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
├── experiments/                    ← runnable run_*.py scripts (setup + eval subcommands)
└── tabflow_slurm/                  ← the package
    ├── __init__.py                 ← re-exports the public API
    ├── run_tabarena_experiment.py  ← runner: fits ONE item on a node (bundled script)
    ├── submit_template.sh          ← sbatch array script (parses job JSON via jq, calls runner)
    ├── slurm_utils.py              ← setup_slurm_job(): per-node Ray init (caches via JobBatch.cache_config)
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
2. `build_setups()` → `_group_jobs()`: each `ModelJob`'s `resources`/`scheduler`/`experiment` dict
   overrides are applied to the base building blocks via `dataclasses.replace`; its `tasks` (a
   `TaskSubset`) is merged onto the plan's `task_subset` (job wins per field). Jobs are **merged**
   when their `(resources, scheduler, task_subset, experiment-with-models-zeroed, ignore_cache)`
   signature matches (compared by value). One `TabArenaBenchmarkSetup` per group.
3. Each setup's `get_jobs_to_run()`: ensure dirs → `experiment_bundle.build_experiments()`
   (attaches each experiment's `ModelConstraints`) → `context.build_jobs(experiments,
   task_subset=...)` (scopes the context's collection by the `TaskSubset`, then enumerates
   experiments × splits; constraint-violating pairs are dropped during enumeration) → scope the
   context's collection to the jobs' tasks and `materialize()` (downloads only those) → Ray cache
   check (core `job_cache_exists_batch` over plain coordinate tuples) → persist the surviving sweep
   as a `JobBatch` artifact (experiments.yaml + task_metadata.csv + jobs.json) →
   `scheduler.bundle_items()`.
4. `scheduler.get_run_commands()` writes the job JSON (splitting at `max_array_size`) and returns the
   `sbatch` command(s). The plan prints one consolidated summary.

Then: `sbatch … submit_template.sh <job.json>` → array task picks `jobs[SLURM_ARRAY_TASK_ID]`,
creates its node-local scratch (below), optionally stages the run's weights and pre-touches the
libraries, then runs `python -P run_tabarena_experiment.py` per item. The runner aligns the thread
variables to the CPU affinity, applies the offline-weights environment, loads the `JobBatch` and the
job (which applies the batch's cache config), and only then calls `setup_slurm_job()`, with Ray
started only when `Experiment.uses_ray` says the fit can reach it; the fit itself goes through
`ExperimentBatchRunner.run_jobs()` (the same execution path as a local benchmark run).
A method whose warm-up raised or left steps failed is aborted before its timed fit (`require_warmup`,
true in the job defaults and the runner CLI); `TabArenaBenchmarkPlan(require_warmup=False)` or
`--require_warmup false` records the report and fits cold instead.

Job scratch layout (`submit_template.sh`): `${TMPDIR:-/tmp}/tj_<SLURM_JOB_ID>/` holds `tmp/`
(the job's `TMPDIR`), `ag/` (`TABARENA_MODEL_ARTIFACTS_BASE_PATH`, the predictor artifacts of every
item), `stage/` (the node-local copy of the weights, `HF_HOME` and `TABPFN_MODEL_CACHE_DIR` when
staging succeeded) and `ray/` (`--ray_temp_root`). The script `cd`s into it before the first
python starts and an `EXIT` trap removes the whole tree. JIT caches (numba, Triton, NVIDIA) live
outside it, keyed per node, user and venv, so later jobs on a warm node reuse them.

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
- **Caches are configured via `CacheConfig` on the context**, not `PathSetup`. The setup embeds
  `context.cache_config` in the `JobBatch`; the head node applies it before materializing tasks and
  each worker applies it (in `run_experiment`) before fitting — so point `CacheConfig.openml` at
  shared storage and the head node + workers resolve the same OpenML cache.
- **Ray on a shared filesystem:** `setup_slurm_job(setup_ray_for_slurm_shared_resources_environment=
  True)` gives each job a unique temp dir + plasma sizing; required when fold-fitting isn't
  sequential-local, else workers collide semi-randomly.
- **Time budget** for `--time` = `ResourcesSetup.time_limit_per_config × configs_per_job +
  time_limit_overhead` (hours). `configs_per_job` is the worst-case bundle size from
  `bundle_items()`.
- **Load before Ray.** The runner loads the batch and the job before `setup_slurm_job()` so that
  `Experiment.uses_ray(problem_type=...)` can decide whether a Ray runtime is started at all; a fit
  that never reaches Ray (a foundation model, a sequential bag) runs without the daemons and their
  0.5 to 1.5 GB RSS. Keep that order when editing the runner, and never write `os.environ` at
  module import time in a model module: the thread variables and the offline-weights environment
  are set once by the runner before any heavy import.
- **`offline_weights`** (`TabArenaBenchmarkPlan`, default `"auto"`): jobs export `HF_HUB_OFFLINE=1`,
  `HF_HUB_DISABLE_PROGRESS_BARS=1` and `AG_FETCH_PRETRAINED_WEIGHTS=false` when the head-node prefetch
  covered every selected model (`PrefetchReport.complete`), no system is selected (a system fetches
  its own checkpoints at fit time) and no run encodes text at fit time. A cache miss is then a failed item, never a download inside the timed fit. The HF cache
  must be shared between the head node and the compute nodes.
- **Node staging defaults** (`NodeStagingSetup` on `SlurmSetup`): `stage_weights=True` copies the
  run's HF `models--*` repo dirs and TabPFN checkpoints (enumerated by
  `tabarena.models.staging.collect_weight_paths` from the prefetchers' return values) into the job
  scratch with an rsync, keeping `reserve_bytes` (10 GB) free and falling back to the shared
  filesystem on any error; non-staged repo dirs are overlaid as symlinks so unenumerated assets still
  resolve. `pretouch_libs=True` reads the job's model libraries into the page cache once per node
  boot (`node_prep.py`, `pretouch_max_bytes` 2 GB). Everything lands in `defaults.staging` of the
  job JSON; a job JSON without the block runs without staging.
- **Continue on failure.** A failing item does not abort its bundle: the template logs `##### item
  FAILED (exit N)`, keeps going (stopping only after three consecutive failures) and exits non-zero
  at the end, so `sacct` still shows `FAILED` while the successful items' `results.pkl` are on disk.
  The runner cleans up after a failed item (`cleanup_on_failure=True`) and copies the Ray worker
  logs of that item to `slurm_out/<benchmark>/<ARRAY_JOB_ID>/ray_logs/task_<i>/`.
- **`fake_memory_for_estimates`** intentionally lies to a model's memory estimator; it does not
  change the SLURM `--mem`. Set it to the GPU's VRAM in GB for every GPU model — AutoGluon budgets
  parallel bagging folds against the reported memory limit (node RAM by default) and never accounts
  VRAM, so without it co-scheduled folds can OOM the GPU. Also usable VRAM-vs-RAM for TFMs.

## Editing tasks

- Changing how jobs are launched/batched → `setup/scheduler.py` (add a `SchedulerSetup` subclass for
  a non-SLURM scheduler; the plan/engine are scheduler-agnostic).
- Changing what runs on a node → `run_tabarena_experiment.py` (Python) and/or `submit_template.sh`
  (bundle iteration); these two move together.
- New hardware preset → a `ResourcesSetup` subclass in `setup/resources.py`.
- The model **selection / config counts / preprocessing / constraints** come from `tabarena`'s
  `TabArenaExperimentBundle`, and the tasks from an **arena context** (`TabArenaContext` /
  `BeyondArenaContext`) scoped by a `TaskSubset`, **not** here — this package only schedules. The
  scope filters are defined once on `tabarena`'s `TaskSubset` (the source of truth for
  `subset_tasks` / `build_jobs`); don't re-declare them in this package.

## Tests & lint

Tests live in the repo's top-level `tests/` dir, under `tests/tabflow_slurm/`; run them with
`pytest tests/tabflow_slurm` (or a bare `pytest` from the repo root for the whole suite). Run
`ruff check .` / `ruff format .` from the repo root (config `../../ruff.toml`;
`from __future__ import annotations` is required). After touching this package, also smoke-test a setup script's
`setup_jobs()` (it runs Ray + materialization locally).
