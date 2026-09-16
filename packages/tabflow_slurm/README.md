# tabflow_slurm

Run TabArena benchmarks on a **SLURM** cluster, or on **SkyPilot** managed jobs.

`tabflow_slurm` turns "I want to fit these models on these tasks with this hardware" into
ready-to-run `sbatch` commands. You compose a **plan** from a few typed building blocks, call
`setup_jobs()`, and it:

1. resolves which `(experiment, dataset, fold, repeat)` work units actually need to run (skipping
   cache hits and units that violate a model's constraints),
2. bundles them into SLURM array tasks,
3. writes a self-contained `JobBatch` artifact + a job JSON, and
4. prints the `sbatch` command(s) to launch.

Each array task then runs one bundled item at a time via a small runner script, caching results
into the workspace where the evaluation code can pick them up.

Swap the scheduler for `SkyPilotSetup` and the same plan runs as SkyPilot managed jobs on GCP VMs
that share nothing with the head node: the venv is rebuilt from a frozen manifest, the bundles are
handed out through a claim queue in a bucket, and the results are mirrored back into the workspace
before the cache check and the evaluation (see "SkyPilot" below).

It is self-contained and only depends on `tabarena` (plus the `sky` CLI for SkyPilot runs).

---

## Install

`tabflow_slurm` is its own package (declared in `pyproject.toml`) that depends on `tabarena`. From
the repo root:

```bash
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"   # tabarena + model fitting
uv pip install -e ./packages/tabflow_slurm                              # this package
```

You also need the cluster to have `jq` available on the compute nodes (the submit script parses the
job JSON with it) and a Python venv reachable from the nodes (passed as `python_path`).

For SkyPilot runs the cluster's `sky` CLI (the PriorLabs fork, installed as a `uv` tool) must be on
`PATH` and connected to the shared API server, which resolves `RTXPRO6000` and picks the regions
(login nodes preset the endpoint; elsewhere export it):

```bash
export SKYPILOT_API_SERVER_ENDPOINT=http://skypilot-api:46580   # only where the login shell does not preset it
sky api info && sky check gcp                                   # server HEALTHY, GCP enabled
```

`SkyPilotSetup` uses that `sky` from `PATH` (`sky_binary` overrides it). Without the cluster CLI the
`tabflow_slurm[skypilot]` extra installs upstream `skypilot[gcp]` next to the run venv's python; that
one runs its own local API server, so set `infra` yourself then. The `gcloud` CLI must be on `PATH`.

---

## Quickstart

A setup script composes a `TabArenaBenchmarkPlan` and calls `setup_jobs()`. Minimal example
(the `setup` half of [`experiments/run_tabarena_v0pt1.py`](experiments/run_tabarena_v0pt1.py)):

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts.tabarena.context import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaBenchmarkPlan(
    benchmark_name="my_benchmark_2026",
    model_jobs=[
        ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),  # GPU model
        ModelJob(models=("Linear", 1), name="cpu"),  # CPU model, 1 random config
    ],
    context=TabArenaContext(),  # owns the tasks + subset predicates (from tabarena)
    task_subset=TaskSubset(subset="lite"),  # typed scope for context.build_jobs (first split only)
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),  # how to build the models (from tabarena)
    path_setup=PathSetup(workspace="/shared/workspace", python_path="/shared/venv/bin/python"),
    resources_setup=TabArenaV0pt1ResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(),
)

plan.setup_jobs()  # prints the sbatch command(s) to launch
```

> **BeyondArena:** swap in `from tabarena.contexts.beyondarena.context import BeyondArenaContext`,
> pass `context=BeyondArenaContext()`, and scope with e.g. `task_subset=TaskSubset(dataset_names=[...])`
> (omit `task_subset` to run the full suite). The context also supplies and asserts the arena's inner
> validation protocol; pair it with `BeyondArenaExperimentBundle` (the TabArena bundle carries no protocol
> of its own either, but the datasets and preprocessing differ).

`TaskSubset` (from `tabarena`) is the typed, single source of truth for the scope filters — the same
fields `TaskMetadataCollection.subset_tasks` / `context.build_jobs` accept (`subset`, `dataset_names`,
`split_indices`, `problem_types`, `n_train_samples`, ...). A plain dict still works (it resolves to a
`TaskSubset`, with unknown keys rejected).

Run the script's `setup` subcommand on the **head node** (it materializes tasks + checks the cache
locally), then run the printed `sbatch` command(s) to launch the jobs. When they finish, evaluate
with the same script's `eval` subcommand (see [`experiments/`](experiments)).

The example scripts take `--scheduler slurm|skypilot|skypilot-pool`; `setup` and `eval` both read it,
so a SkyPilot run is launched and evaluated with the same two commands.

---

## End-to-end flow

```
                          your setup script (experiments/run_*.py setup)
                                          │  composes
                                          ▼
   PathSetup ─┐                  TabArenaBenchmarkPlan ──── ModelJob[] (per-model overrides)
   ResourcesSetup ─┤  building     │  setup_jobs()
   SchedulerSetup ─┤  blocks       │
   arena context ──┤ (from         ├─ (optional) prefetch foundation-model weights on this node
   experiment bundle) tabarena)    ├─ group ModelJobs by effective settings  ──►  one
   (+ TaskSubset)                  │                                            TabArenaBenchmarkSetup
                                   │                                            per group (internal)
                                   ▼
                       per group: build experiments → context.build_jobs
                       (experiments × splits, scoped by the TaskSubset;
                       constraint violations dropped) → materialize the jobs'
                       tasks → Ray-filter cache hits → write JobBatch artifact
                       → bundle into array tasks → write job JSON
                                   │
                                   ▼
                       prints:  sbatch --array=0-K%N … submit_template.sh <job.json>
                                          │  you run this
                                          ▼
                       SLURM array job: each task runs submit_template.sh
                                   │  reads its bundle from <job.json> (via jq)
                                   ▼
                       run_tabarena_experiment.py  (once per item in the bundle)
                                   │  setup_slurm_job() → JobBatch.load() →
                                   │  ExperimentBatchRunner.run_jobs() → cache result
                                   ▼
                       <workspace>/output/<benchmark_name>/…   ──►  run_*.py eval → leaderboard
```

With `SkyPilotSetup` the lower half becomes:

```
                       prints:  sky check gcp
                                sky jobs launch -y -d -n <launch_id> --num-jobs N <job.yaml>
                                (pool mode: sky jobs pool apply / pool status first, pool down after)
                                          │  you run this
                                          ▼
                       N managed jobs, each a spot VM (or a pool worker):
                       setup: build the venv from the staged env.json manifest
                       run:   python -P -m tabflow_slurm.sky_worker
                                   │  claims a bundle (exclusive create in <bucket>/.../queue/<launch_id>/claims/)
                                   ▼
                       run_tabarena_experiment.py --cache_root ... --materialize_tasks True  (per item)
                                   │  downloads the dataset through the batch's recorded suite, fits, and
                                   │  copies data/<method>/<task>/<r_f>/ into <bucket>/.../output/data/
                                   ▼
                       run_*.py eval  ──►  sync_results_to_local (bucket → workspace)  ──►  leaderboard
```

---

## The building blocks

The package is `tabflow_slurm/` with a `setup/` subpackage. `TabArenaBenchmarkPlan` is the **single
public entry point**; everything below is re-exported from the top-level package for convenience.

### `TabArenaBenchmarkPlan` — `setup/plan.py`
The thing you construct. A base setup (paths / resources / scheduler / **arena context** /
experiment) plus an optional plan-level `task_subset` (a `TaskSubset`) and a list of `ModelJob`s.
`setup_jobs()`:
- optionally **prefetches foundation-model weights** for the selected models on the head node
  (`prefetch_model_weights=True`), so offline/parallel compute nodes find them cached;
- **groups** `ModelJob`s whose *effective* `(resources, scheduler, task_subset,
  experiment-minus-models, ignore_cache)` are identical into one run — so e.g. a GPU job and a CPU
  job become two separate `sbatch` commands automatically;
- builds one internal `TabArenaBenchmarkSetup` per group, runs it, and prints **one consolidated
  summary** plus all the commands to launch.

### `ModelJob` / `SingleModel` — `setup/plan.py`
- **`ModelJob`** — one or more models that share a set of per-job **overrides** on the base setup:
  `resources` / `scheduler` / `experiment` (dicts applied with `dataclasses.replace`; unknown keys
  raise) plus `tasks` (a `TaskSubset` — or dict — merged onto the plan's `task_subset`, the job
  winning per field). `name` labels the group's `parallel_benchmark_name`; `ignore_cache` forces a
  rerun.
- **`SingleModel`** — the typed form of the `(name, n_configs)` tuples that experiment bundles
  accept. `n_configs`: `int` (that many random configs; `0` = default only), `"all"`, or a `dict`
  (AutoGluon full-pipeline kwargs). Pre-built `Experiment` objects can also be passed and pass
  through untouched.

### `PathSetup` — `setup/paths.py`
All paths, derived from a single **`workspace`** dir + the **`python_path`** for the jobs. Inside the
workspace it creates and uses:

| Subdir | Contents |
| --- | --- |
| `output/<benchmark_name>/` | benchmark result artifacts (the cache the runner writes / eval reads) |
| `slurm_out/<benchmark_name>/` | SLURM `.out` logs |
| `setup_out/<benchmark_name>/` | generated `JobBatch` artifact(s) + job JSON |

Cache locations (OpenML / HuggingFace / TabArena) are **not** part of `PathSetup` — configure them
on the arena context via `TabArenaContext(cache_config=CacheConfig.from_root(...))`, exactly as in
the sequential/async API. The plan embeds that `CacheConfig` in the `JobBatch`, and each worker
applies it automatically (see `tabarena.caching.CacheConfig`).

`run_script` / `submit_script` default to the scripts **bundled with this package**
(`get_run_script_path()` / `get_submit_script_path()`), so nothing hardcodes a checkout path.

### `ResourcesSetup` — `setup/resources.py`
Compute + time budget per fit: `time_limit` (s), `num_cpus`, `num_gpus`, `memory_limit` (GB), plus:
- `num_gpus_model` — GPUs given to the *model* (set `0` to reserve the GPU for preprocessing, e.g.
  sentence-transformer text encoding, while fitting on CPU);
- `fake_memory_for_estimates` — report a different memory figure to a model's internal estimator
  (e.g. when VRAM ≫ host RAM for foundation models) — experimental;
- preprocessing-time knobs (`time_limit_*_model_agnostic_preprocessing`).

Presets: **`TabArenaV0pt1ResourcesSetup`** (8 CPU / 32 GB / 1h) and **`BeyondArenaResourcesSetup`**
(auto CPU/RAM, 4h; docstring records the exact GPU/CPU node specs used).

### `SchedulerSetup` / `SlurmSetup` / `GCPSlurmSetup` — `setup/scheduler.py`
- **`SchedulerSetup`** (base) owns scheduler-agnostic **batching**: `bundle_size` (items per array
  task) and `bundle_size_per_dataset` (per-dataset override). `bundle_items()` groups the approved
  jobs by effective bundle size (dataset shapes looked up in the task collection) into
  `{"items": [...]}` array tasks.
- **`SlurmSetup`** turns the job dict into JSON file(s) + `sbatch` command(s): GPU/CPU partitions,
  `extra_gres`, `exclusive_node`, memory style (`mem_per_handle`), `time_limit_overhead`,
  `array_job_limit` (the `%N` concurrency cap), and `max_array_size` (splits very large arrays into
  multiple batches/commands). The per-task `--time` is budgeted from
  `time_limit_per_config × configs_per_job + overhead`.
- **`GCPSlurmSetup`** — the BeyondArena GCP defaults (partition names, `exclusive_node=True`).

Two hooks on the base class let a scheduler without a shared filesystem plug in: `sync_results_to_local`
(called by the engine before the cache check and by the scripts before an eval; a no-op for SLURM) and
`prefetches_weights_on_head` (False when the nodes fetch model weights themselves, which skips the
head-node prefetch). `describe_target` names where a run lands for the plan's banner.

### `SkyPilotSetup` — `setup/skypilot.py`
Runs the bundles as SkyPilot managed jobs draining a **GCS claim queue**. Construct it like a
`SlurmSetup`; `PathSetup` keeps naming the head node's workspace and venv. `setup_jobs()` then:
- **replicates the run venv** (`setup/sky_env.py`): `uv pip freeze` of `python_path`, a content-addressed
  archive of every local checkout it installs (this repo, the AutoGluon fork, local sdists), and an
  `env.json` manifest under `<bucket>/<prefix>/env/<hash>/`; the worker's `setup:` rebuilds an
  identical venv from it (working tree, not `HEAD`, so uncommitted fixes ship too);
- copies the `JobBatch` and **one task JSON per bundle** to `<bucket>/<prefix>/runs/<benchmark>/queue/<launch_id>/`;
- renders `job.yaml` (and `pool_<name>.yaml` with `use_pool=True`) under `setup_out/<benchmark>/sky/`
  and prints **one `sky jobs launch -y -d --num-jobs N`** per run group (never one launch per bundle).

Knobs: `bucket` / `prefix` (defaults to the org's EU sky-cache bucket and `<user>/tabarena`),
`api_server_endpoint` (printed as an export; unset means the shell must provide it), `infra` (unset by
default: the shared server's admin policy expands the regions and rejects an explicit one), `workers`
(concurrent worker jobs, the `%N` analogue, and the pool size), `use_pool` / `pool_name`,
`gpu_accelerator` (`RTXPRO6000:1`, a `g4-standard-48` with 96 GB VRAM, 48 vCPU, 180 GB RAM; pair with
`fake_memory_for_estimates=96`),
`cpu_cpus` / `cpu_memory` / `cpu_instance_type`, `use_spot`, `disk_size`, `item_time_limit_overhead`
(seconds per **item**, the worker kills an item over budget and its bundle mates still run), `secrets`
(names forwarded from the shell, e.g. `HF_TOKEN`), `requirements_extra_lines`, `sky_binary`.

The worker (`sky_worker.py`) resumes the bundles its job already owns after a preemption (SkyPilot keeps
`SKYPILOT_TASK_ID` across recoveries), then claims unowned bundles until nothing is left, writing
`done/<bundle>.<item>` and `failed/...` markers the progress watcher counts. Each item runs the bundled
runner with `--cache_root` (every cache under one local directory) and `--materialize_tasks True` (the
dataset is downloaded through the suite the batch recorded in `task_source.json`, OpenML for TabArena
and data-foundry for BeyondArena), so no shared cache is needed. Results are copied per item into
`runs/<benchmark>/output/data/`; `sync_results_to_local` mirrors that prefix into
`<workspace>/output/<benchmark>/data` (nothing deleted, so SLURM and SkyPilot results merge).

### `TabArenaBenchmarkSetup` — `setup/benchmark.py` *(internal)*
The per-run engine for one homogeneous run. Not part of the public API — the plan builds and drives
it. `get_jobs_to_run()` is the core pipeline: ensure dirs → build the experiments (the bundle
attaches each experiment's `ModelConstraints`) → `context.build_jobs(experiments,
task_subset=...)` (scopes the context's collection by the `TaskSubset`, then enumerates experiments
× splits; constraint-violating pairs are dropped during enumeration) → scope the context's
collection to the jobs' tasks and `materialize()` them (download only those) → Ray cache check
(tabarena core's writer-aligned `job_cache_status_batch`, fanned out over plain
`(method, task_id_str, fold, repeat, protocol_key)` tuples; a cached result fit under another
validation protocol stops the setup) → persist the surviving sweep as a self-contained
`JobBatch` artifact (experiments.yaml + task_metadata.csv + jobs.json + the context's
validation_protocol.json, which the compute node re-checks before fitting) → bundle.

### Runtime (what runs on the node)
- **`run_tabarena_experiment.py`** — the runner a single array task invokes per item (with
  `python -P`). Aligns the thread variables to the CPU affinity, applies the offline-weights
  environment, loads the shipped `JobBatch`, looks the item's `(experiment, dataset, fold, repeat)`
  coordinates up in the batch's serialized job list (stale coordinates fail loudly), then calls
  `setup_slurm_job()` (Ray is started only when the experiment's fit can reach it) and runs the job
  through `ExperimentBatchRunner.run_jobs` — the exact same execution path (task resolution,
  results naming, cache layout) as a local benchmark run. Results cache under
  `output/<benchmark_name>/`. A failed item is cleaned up and its Ray worker logs are copied next to
  the SLURM output.
- **`submit_template.sh`** — the `sbatch` array script. Creates a per-job node-local scratch
  directory (`${TMPDIR:-/tmp}/tj_<job id>/` with `tmp/`, `ag/` for the predictor artifacts, `stage/`
  for staged weights, `ray/`), optionally copies the run's foundation-model weights into it and
  pre-touches the model libraries (`defaults.staging`), reads the array index's `items` from the job
  JSON (with `jq`) and runs the runner once per item. A failing item is logged (`##### item FAILED`)
  and its siblings still run; the task exits non-zero at the end. An `EXIT` trap removes the scratch
  directory.
- **`slurm_utils.py::setup_slurm_job`** — per-node setup: initializes Ray for a **shared-filesystem**
  SLURM environment (unique temp dir under the job scratch, plasma store sizing, forkserver) so
  parallel workers don't collide. (Caches are configured separately, by `run_experiment` applying
  the `JobBatch`'s `cache_config`.)
- **`node_prep.py`** — the once-per-node-boot page-cache pre-touch of the job's libraries, invoked by
  the template.
- **`sky_worker.py`** — the SkyPilot counterpart of `submit_template.sh`: one process per worker job
  that drains the claim queue and runs the runner per item (argument list from
  `run_local._build_item_command`, plus `--cache_root` and `--materialize_tasks`), under a per-item
  budget, uploading results and logs to the bucket.
- The runner's two opt-in flags for nodes without the shared filesystem: `--cache_root <dir>` (every
  cache under one local directory instead of the batch's head-node `cache_config`) and
  `--materialize_tasks True` (download this job's dataset through the suite recorded in the batch's
  `task_source.json` before fitting). Both are off for SLURM and local runs.

---

## The job JSON

`setup_jobs()` writes one (or more) JSON files under `setup_out/<benchmark_name>/`:

```json
{
  "defaults": {
    "python": "/shared/venv/bin/python",
    "run_script": ".../run_tabarena_experiment.py",
    "job_batch_dir": ".../job_batch_<name>",
    "output_dir": ".../output/<benchmark_name>",
    "num_cpus": 8, "num_gpus": 0, "memory_limit": 32,
    "ignore_cache": false,
    "offline_weights": true,
    "setup_ray_for_slurm_shared_resources_environment": true,
    "slurm_log_dir": ".../slurm_out/<benchmark_name>",
    "staging": {"stage_weights": true, "hf_repo_dirs": ["..."], "tabpfn_files": [], "reserve_bytes": 10737418240,
                "pretouch_libs": true, "pretouch_packages": ["..."], "pretouch_max_bytes": 2147483648,
                "jit_cache_max_mb": 2048}
  },
  "jobs": [
    {"items": [{"experiment": "LightGBM_c1", "dataset": "anneal", "fold": 0, "repeat": 0}, ...]},
    ...
  ]
}
```

`SLURM_ARRAY_TASK_ID` selects `jobs[i]`; the submit script runs the runner once per `items` entry,
using `defaults` for everything shared. `offline_weights` (decided by the plan from the head-node
prefetch) makes the jobs resolve checkpoints from the shared cache only, so a miss fails the item
instead of downloading inside the timed fit; `staging` (from `NodeStagingSetup`; the weight copy is
opt-in, the library pre-touch is on) lists the weight files copied onto node-local scratch and the
pre-touch budget. Older job JSONs without these keys still run.

`SkyPilotSetup` writes the same content as one file per bundle, `queue/<launch_id>/tasks/<i>.json`
holding `{"defaults": {..., "item_timeout_seconds": T}, "items": [...]}`. The worker ignores the four
head-node paths in `defaults` (`python`, `run_script`, `job_batch_dir`, `output_dir`) and substitutes
its own; everything else is passed through unchanged.

---

## Examples & history

- [`experiments/`](experiments) — runnable scripts, each with `setup` + `eval` subcommands sharing
  one `benchmark_name` + paths: `run_tabarena_v0pt1.py`, `run_beyondarena.py` (+ `_local`, no-SLURM
  variants). Copy one and adapt the `workspace` / `python_path` / models; run `<script> setup`, then
  `<script> eval` when the jobs finish.
- [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md) — an append-only, newest-first record of real runs. Each
  entry is a **frozen snapshot** of the `setup_jobs()` call + its git SHA (the API evolves, so old
  snippets only run against their recorded commit).

## Notes & gotchas

- **Run setup on the head node.** It materializes tasks (downloading data-foundry datasets into the
  OpenML cache), checks the cache, and prefetches foundation weights — all locally — before any
  `sbatch`.
- **The OpenML cache must be shared (SLURM).** Tasks materialized during setup must be visible to the
  workers. Point `CacheConfig.openml` at shared storage (via `TabArenaContext(cache_config=...)`);
  the config is embedded in the `JobBatch` and applied identically on the head node and every worker.
  SkyPilot workers have no shared cache: they run with `--cache_root` and `--materialize_tasks`, which
  needs the batch's `task_source.json` (written by every setup since the preset is recorded; an older
  batch cannot materialize data-foundry tasks on a VM).
- **SkyPilot: mind the endpoint, the card and the pool.** The `sky` commands need the shared API
  server (`SKYPILOT_API_SERVER_ENDPOINT`, preset on login nodes); the printed block fails fast without
  it. The default card is the same RTX PRO 6000 as the SLURM partition, so `fake_memory_for_estimates`
  stays 96. Gated weights need their token in `secrets`. In pool mode run
  `sky jobs pool down -y <pool>` when the benchmark is finished. `sky jobs cancel -n <launch_id> -y`
  stops a launch; its orphaned claims are re-enumerated by the next `setup` into a fresh queue.
- **Do not re-run `setup` for a `benchmark_name` while its launch is still draining.** The running
  launch is not harmed (a new launch gets its own id and queue), but the head node's cache check only
  sees results that were synced, so items still in flight are enumerated again and fitted twice.
  `setup` checks the bucket and prints a warning (also into the command block) naming any launch of
  the benchmark with bundles not yet done; wait for `sky_progress.sh` to report `DONE` or
  `WORKERS GONE`, or cancel the old launch first. Editing the checkout, the venv or the workspace
  while a launch runs is safe: the workers use the staged environment and batch from the bucket.
- **Grouping is by *effective* settings.** Two `ModelJob`s with the same resources/scheduler/tasks/
  experiment merge into one run (one `JobBatch`, one array). Different `num_gpus` (or any override)
  splits them — that's how GPU vs CPU models become separate `sbatch` commands.
- **Re-running is cache-aware.** A second `setup_jobs()` only emits the items still missing from
  `output/<benchmark_name>/`; pass `ignore_cache=True` (on a `ModelJob`) to force a rerun.
- **A `FAILED` array task is a bundle with at least one failed item**, not a lost bundle: the other
  items' `results.pkl` are on disk, the log names each failed item, and a second `setup_jobs()`
  re-emits only what is missing.
- **Run setup and `run_local` from the repo root.** Ray puts the driver's cwd on every worker's
  `sys.path`; a cwd that contains an `autogluon/` checkout shadows the installed AutoGluon.
