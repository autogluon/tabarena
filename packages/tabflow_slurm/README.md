# tabflow_slurm

Run TabArena benchmarks on a **SLURM** cluster.

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

It is self-contained and only depends on `tabarena`.

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

---

## Quickstart

A setup script composes a `TabArenaBenchmarkPlan` and calls `setup_jobs()`. Minimal example
(see [`experiments/run_setup_tabarena_v0pt1.py`](experiments/run_setup_tabarena_v0pt1.py)):

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabflow_slurm import (
    GCPSlurmSetup, ModelJob, PathSetup, TabArenaBenchmarkPlan, TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaBenchmarkPlan(
    benchmark_name="my_benchmark_2026",
    model_jobs=[
        ModelJob(models=("TabPFN-3", 0), name="gpu", resources={"num_gpus": 1}),  # GPU model
        ModelJob(models=("Linear", 1), name="cpu"),                                # CPU model, 1 random config
    ],
    tasks=TaskMetadataCollection.from_preset("TabArena-v0.1-lite"),  # which tasks (from tabarena)
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),      # how to build the models (from tabarena)
    path_setup=PathSetup(workspace="/shared/workspace", python_path="/shared/venv/bin/python"),
    resources_setup=TabArenaV0pt1ResourcesSetup(),
    scheduler_setup=GCPSlurmSetup(),
)

plan.setup_jobs()   # prints the sbatch command(s) to launch
```

Run the script on the **head node** (it materializes tasks + checks the cache locally), then run the
printed `sbatch` command(s) to launch the jobs. When they finish, evaluate with a
`run_eval_*.py` script (see [`experiments/`](experiments)).

---

## End-to-end flow

```
                          your setup script (experiments/run_setup_*.py)
                                          │  composes
                                          ▼
   PathSetup ─┐                  TabArenaBenchmarkPlan ──── ModelJob[] (per-model overrides)
   ResourcesSetup ─┤  building     │  setup_jobs()
   SchedulerSetup ─┤  blocks       │
   task collection ┤ (from         ├─ (optional) prefetch foundation-model weights on this node
   experiment bundle) tabarena)    ├─ group ModelJobs by effective settings  ──►  one
                                   │                                            TabArenaBenchmarkSetup
                                   │                                            per group (internal)
                                   ▼
                       per group: materialize tasks → build experiments →
                       build_jobs (experiments × splits) → filter constraint
                       violations → Ray-filter cache hits → write JobBatch
                       artifact → bundle into array tasks → write job JSON
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
                       <workspace>/output/<benchmark_name>/…   ──►  run_eval_*.py → leaderboard
```

---

## The building blocks

The package is `tabflow_slurm/` with a `setup/` subpackage. `TabArenaBenchmarkPlan` is the **single
public entry point**; everything below is re-exported from the top-level package for convenience.

### `TabArenaBenchmarkPlan` — `setup/plan.py`
The thing you construct. A base setup (paths / resources / scheduler / tasks / experiment) plus a
list of `ModelJob`s. `setup_jobs()`:
- optionally **prefetches foundation-model weights** for the selected models on the head node
  (`prefetch_model_weights=True`), so offline/parallel compute nodes find them cached;
- **groups** `ModelJob`s whose *effective* `(resources, scheduler, tasks, experiment-minus-models,
  ignore_cache)` are identical into one run — so e.g. a GPU job and a CPU job become two separate
  `sbatch` commands automatically;
- builds one internal `TabArenaBenchmarkSetup` per group, runs it, and prints **one consolidated
  summary** plus all the commands to launch.

### `ModelJob` / `SingleModel` — `setup/plan.py`
- **`ModelJob`** — one or more models that share a set of per-job **overrides** (dicts) on the base
  setup: `resources`, `scheduler`, `tasks`, `experiment`. `name` labels the group's
  `parallel_benchmark_name`; `ignore_cache` forces a rerun. Overrides are applied with
  `dataclasses.replace`, and unknown keys raise (listing valid field names).
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
| `.openml-cache/` | OpenML cache (unless `openml_cache` overridden; `"auto"` uses OpenML's default) |

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

### `TabArenaBenchmarkSetup` — `setup/benchmark.py` *(internal)*
The per-run engine for one homogeneous run. Not part of the public API — the plan builds and drives
it. `get_jobs_to_run()` is the core pipeline: ensure dirs → materialize the task collection →
build the experiments (the bundle attaches each experiment's `ModelConstraints`) → core
`build_jobs` (experiments × splits; constraint-violating pairs are dropped during enumeration) →
Ray cache check (tabarena core's writer-aligned `job_cache_exists_batch`, fanned out over plain
`(method, task_id_str, fold, repeat)` tuples) → persist the surviving sweep as a self-contained
`JobBatch` artifact (experiments.yaml + task_metadata.csv + jobs.json) → bundle.

### Runtime (what runs on the node)
- **`run_tabarena_experiment.py`** — the runner a single array task invokes per item. Loads the
  shipped `JobBatch`, looks the item's `(experiment, dataset, fold, repeat)` coordinates up in the
  batch's serialized job list (stale coordinates fail loudly), calls `setup_slurm_job()`, then runs
  that job through `ExperimentBatchRunner.run_jobs` — the exact same execution path (task
  resolution, results naming, cache layout) as a local benchmark run. Results cache under
  `output/<benchmark_name>/`.
- **`submit_template.sh`** — the `sbatch` array script. Reads `defaults` + the array index's `items`
  from the job JSON (with `jq`) and runs the runner once per item.
- **`slurm_utils.py::setup_slurm_job`** — per-node setup: points OpenML at the right cache and
  initializes Ray for a **shared-filesystem** SLURM environment (unique temp dir, plasma store
  sizing, forkserver) so parallel workers don't collide.

---

## The job JSON

`setup_jobs()` writes one (or more) JSON files under `setup_out/<benchmark_name>/`:

```json
{
  "defaults": {
    "python": "/shared/venv/bin/python",
    "run_script": ".../run_tabarena_experiment.py",
    "openml_cache_dir": ".../.openml-cache",
    "job_batch_dir": ".../job_batch_<name>",
    "output_dir": ".../output/<benchmark_name>",
    "num_cpus": 8, "num_gpus": 0, "memory_limit": 32,
    "ignore_cache": false,
    "setup_ray_for_slurm_shared_resources_environment": true
  },
  "jobs": [
    {"items": [{"experiment": "LightGBM_c1", "dataset": "anneal", "fold": 0, "repeat": 0}, ...]},
    ...
  ]
}
```

`SLURM_ARRAY_TASK_ID` selects `jobs[i]`; the submit script runs the runner once per `items` entry,
using `defaults` for everything shared.

---

## Examples & history

- [`experiments/`](experiments) — runnable setup + eval scripts:
  `run_setup_tabarena_v0pt1.py`, `run_setup_beyondarena.py`, `run_eval_tabarena_v0pt1.py`,
  `run_eval_beyondarena.py`. Copy one and adapt the `workspace` / `python_path` / models.
- [`BENCHMARK_LOG.md`](BENCHMARK_LOG.md) — an append-only, newest-first record of real runs. Each
  entry is a **frozen snapshot** of the `setup_jobs()` call + its git SHA (the API evolves, so old
  snippets only run against their recorded commit).

## Notes & gotchas

- **Run setup on the head node.** It materializes tasks (downloading data-foundry datasets into the
  OpenML cache), checks the cache, and prefetches foundation weights — all locally — before any
  `sbatch`.
- **The OpenML cache must be shared.** Tasks materialized during setup must be visible to the
  workers; they configure the same cache via `--openml_cache_dir`.
- **Grouping is by *effective* settings.** Two `ModelJob`s with the same resources/scheduler/tasks/
  experiment merge into one run (one `JobBatch`, one array). Different `num_gpus` (or any override)
  splits them — that's how GPU vs CPU models become separate `sbatch` commands.
- **Re-running is cache-aware.** A second `setup_jobs()` only emits the items still missing from
  `output/<benchmark_name>/`; pass `ignore_cache=True` (on a `ModelJob`) to force a rerun.
