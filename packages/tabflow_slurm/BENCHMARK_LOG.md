# Benchmark Log

A historical record of benchmark setups run on the SLURM cluster. **Each entry is a
frozen snapshot** of the exact `setup_jobs()` call used at that time — it is *not*
kept in sync with the current `tabflow_slurm` API and is not expected to import or
run against `main`. To reproduce an entry, check out its recorded **git SHA**.

## Conventions

- **Append-only, newest-first.** Add new runs at the top of the log below.
- **Never refactor old code blocks.** They document what was literally run; the API
  changes over time (`BenchmarkSetup` → `BenchmarkSetup2026` → `TabArenaBenchmarkPlan`),
  so a snippet only makes sense against the commit it ran on.
- **Always record the git SHA** (`git -C tabarena rev-parse --short HEAD`). It is the
  only thing that makes an old snippet reproducible.
- Keep the *why* (config counts, partition choices, constraints) in the notes — that's
  the context that isn't recoverable from the code alone.

### Entry template

```markdown
## YYYY-MM-DD — <benchmark_name>

- **Model(s):** <Model> (<n_configs>)
- **Git SHA:** `<short-sha>`
- **Purpose:** <one line>
- **Notes:** <partition, constraints, runtime, anything non-obvious>

​```python
<verbatim setup_jobs() call as run>
​```
```

---

## 2026-09-13 — aplr_10092026 (time-limit rerun)

- **Model(s):** APLR (all configs; the 1255 items still missing)
- **Git SHA:** `346c1508` (PR #515 head) plus uncommitted local edits in `models/aplr/`: `info.py`
  `verified=True`, `model.py` with `_estimate_memory_usage_static` and, new here, forwarding of the
  AutoGluon fit budget to aplr's `time_limit` (see notes).
- **Purpose:** Finish the run with the 3600 s per-config budget actually enforced, after the two
  earlier arrays let APLR boost past it.
- **Notes:** Investigation of the SDSS17 timeouts: the budget is only a hint below SLURM. TabArena
  injects `ag.max_time_limit=3600`, AutoGluon's bagged model hands each fold a share (about 360 s
  when SDSS17 fits one fold at a time), but its Ray fold strategy never checks the clock after a
  fold and `AbstractModel.fit` never compares fit time against the limit; the wrapper then dropped
  `time_limit` on the floor and aplr 10.26.0 has no budget knob at all (runtime is set by `m=3000`
  and early stopping). 41 % of the 1161 stored SDSS17 results exceeded 1 h (median 0.90 h, p90
  7.2 h, max 15.3 h); the driver is `min_observations_in_split`: values up to 0.3 take 7 to 9 h,
  values above 0.4 about 0.8 h. Fix in two parts. (1) aplr fork
  `LennartPurucker/aplr`, branch `time-limit` (commit `3eba5db`, upstream PR
  https://github.com/ottenbreit-data-science/aplr/pull/18): a `time_limit` constructor parameter
  (seconds, NaN = off) that is split evenly across the inner cv folds and the one-vs-rest logit
  models, with boosting aborting after the step that exhausts a fold's share; installed into the run
  venv `tabarena_10082026` from the local clone (`uv pip install --python <venv> <clone>`, version
  string still 10.26.0). (2) The wrapper passes 95 % of the time left after preprocessing as
  `time_limit` and warns when the installed aplr lacks the parameter. Before relaunching, every
  stored result with `time_train_s > 3600` (552: 480 SDSS17, 72 kddcup09_appetency, found by
  reading all 163313 `results.pkl`) was moved, not deleted, to
  `output/aplr_10092026_over_time_limit/data/` with `manifest.csv` next to it. The earlier cleanup
  array `1149802` had ended with 2148 completed / 226 SIGTERM / 477 cancelled. `setup` then
  enumerated 1255 items (SDSS17 1128, kddcup09_appetency 84, superconductivity 20, a 5-dataset
  tail) into 126 ten-item tasks and the array was launched 2026-09-13 as job `1153217`
  (`--array=0-125%200`, `--time=16:00:00`), back on `bundle_size=10` with `time_limit_overhead=6`
  since a bundle can no longer exceed 10 h of fitting. Partition `cpun416mtspotinteractive`. Its
  first SDSS17 configs finished in 2161 s and 2194 s (one fold at a time), no fit over 3600 s in
  any log. After 926 items the maintainer asked to use the full 300-slot allowance, so job
  `1153217` was cancelled (one task had been preempted) and the 329 leftover items relaunched the
  same day as job `1153519` with `bundle_size=1, time_limit_overhead=2, array_job_limit=300`
  (`--array=0-328%300`, `--time=3:00:00`); the code block below shows that final call. It ended
  319 completed / 10 preempted; the ten leftovers ran as job `1154056` (all completed), after
  which `setup` approved 0 items and `data/` held 164016 `results.pkl` (201 configs x 816 splits).
  No fit in any log of the three time-limited arrays exceeded 3600 s. Processed and uploaded as
  suite `tabarena-2026-09-13` via `tabarena.models.aplr.info:aplr_method_metadata`
  (`config_default="aplr_c1_default_BAG_L1"`, r2 bucket `tabarena`, prefix `cache`).

```python
def setup() -> None:
    """Generate the job JSON and emit the ``sbatch`` command(s) for the run."""
    plan = TabArenaV0pt1BenchmarkPlan(
        benchmark_name=BENCHMARK_NAME,
        model_jobs=[
            ModelJob(models=(MODEL, NUM_CONFIGS), name="cpu"),
        ],
        task_subset=TaskSubset(),  # all splits of every task
        path_setup=_path_setup(),
        experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
        resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
        # Same CPU partition as the ChimeraBoost / CTBoost runs (16 vCPUs, 64 GB RAM) for comparable
        # timings; 200 array tasks at once instead of the default 100.
        scheduler_setup=GCPSlurmSetup(
            # Restart of the time-limit rerun (job 1153217, ten-item bundles, %200) with the whole
            # 300-slot allowance: about 330 items were left, so one item per task fills every slot
            # and a spot preemption loses a single item. APLR now honors the 3600 s budget through
            # aplr's `time_limit` (fork branch time-limit, upstream PR #18), so 1 h + 2 h overhead
            # covers an SDSS17 config with its 8 sequential folds.
            bundle_size=1,
            time_limit_overhead=2,
            cpu_partition="cpun416mtspotinteractive",
            array_job_limit=300,
        ),
    )
    plan.setup_jobs()
```

---

## 2026-09-10 — aplr_10092026 (full run)

- **Model(s):** APLR (all configs)
- **Git SHA:** `346c1508` (PR #515 head) plus two uncommitted local edits in `models/aplr/`: `info.py`
  `verified=True`, and `model.py` gains `_estimate_memory_usage_static` (see notes).
- **Purpose:** Full TabArena run of APLR after the Lite pass below (same key; `setup` skipped the 1246
  cached Lite items and enumerated the remaining 162770).
- **Notes:** The Lite pass OOM-killed every config of Amazon_employee_access, kddcup09_appetency and
  SDSS17: APLR one-hot encodes categoricals into a dense float64 matrix inside its C++ core (about
  7 copies of `n_rows x one_hot_width` for binary/regression, `4.5 + 3.6 * n_classes` copies for
  multiclass; measured 7.9 GB, 26.4 GB and 28.6 GB per fold on those three), and without a memory
  estimate AutoGluon fitted 8 folds in parallel on the 62 GB nodes. The new estimate makes AutoGluon
  fit Amazon with 4 folds in parallel and kddcup09 / SDSS17 one fold at a time; all other tasks stay
  at 8. First launch 2026-09-10 23:06 as two arrays (SDSS17 one item per task, job `1129137`;
  the rest in bundles of 10, job `1129138`); both stopped overnight with about 69k items done.
  Relaunched 2026-09-11 11:48 as ONE array, job `1138562` (93505 remaining items in 9351 ten-item
  tasks, `--array=0-9350%200`, `--time=16:00:00`): the maintainer wants a single array per run, and
  `time_limit_overhead=6` gives a ten-item SDSS17 bundle (8 sequential folds per config, a single
  fold took 16 min on 2 threads, APLR ignores the per-config time limit) room to finish. Partition
  `cpun416mtspotinteractive` (16 vCPUs, 64 GB RAM), run venv `tabarena_10082026` with
  `aplr==10.26.0`. Job `1138562` drained on 2026-09-12 with 8912 completed / 309 failed (spot
  SIGTERMs, plus 9 Marketing_Campaign tasks that hit a stale NFS handle on the OpenML cache file) /
  130 timed out (all SDSS17: its configs take ~4 h at the median with sequential folds and the slow
  tail exceeds 16 h). No OOM in any of the 9351 logs. Cleanup pass for the 2851 missing items
  (1104 SDSS17, 341 kddcup09_appetency, the rest a preemption tail over 20 datasets) launched
  2026-09-12 as job `1149802`: same plan with `bundle_size=1, time_limit_overhead=23`
  (`--array=0-2850%200`, `--time=24:00:00`), so a preemption loses one item and SDSS17 configs get
  a day each.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="aplr_10092026",
    model_jobs=[
        ModelJob(models=("APLR", "all"), name="cpu"),
    ],
    task_subset=TaskSubset(),  # all splits of every task
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(
        bundle_size=10,
        time_limit_overhead=6,
        cpu_partition="cpun416mtspotinteractive",
        array_job_limit=200,
    ),
)
plan.setup_jobs()
```

---

## 2026-09-10 — aplr_10092026

- **Model(s):** APLR (default + 25 configs)
- **Git SHA:** `346c1508` (PR #515 head, fork branch `mathias-von-ottenbreit/tabarena:main` merged with `main`)
- **Purpose:** First TabArena-Lite pass of APLR (Automatic Piecewise Linear Regression,
  https://github.com/ottenbreit-data-science/aplr), integrated in
  https://github.com/autogluon/tabarena/pull/515, to gauge accuracy and runtime before a full run.
- **Notes:** Lite subset (first split of every task), default config + the first 25 configs of the
  frozen HPO portfolio. Same CPU partition as the ChimeraBoost / Perpetual / CTBoost runs,
  `cpun416mtspotinteractive` (16 vCPUs, 64 GB RAM), `memory_limit`/`num_cpus` left `None` so node
  values are picked up, bundle size 2. A later full run can reuse this key and skip the cached items.
  `array_job_limit=200` (default 100) so up to 200 array tasks run at once. Extra dep in the run
  venv: `aplr==10.26.0` (`tabarena_10082026`). No `fake_memory_for_estimates` (CPU model).
  `APLRModel` implements no `_estimate_memory_usage_static` and no `warmup`, and its `_fit` discards
  `time_limit`, so a slow config runs until it finishes or SLURM's `--time` kills the array task
  (both bundled items are then lost; a `setup` rerun re-enumerates them under the same key). The
  registry's `MethodMetadata` (`suite="tabarena-2026-09-04"`) predates the run; fix the date when
  uploading. Launched 2026-09-10 19:33 as SLURM array `1127887` (1326 items in 663 tasks,
  `--array=0-662%200`, `--time=3:00:00`).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="aplr_10092026",
    model_jobs=[
        ModelJob(models=("APLR", 25), name="cpu"),
    ],
    task_subset=TaskSubset(subset="lite"),  # first split of every task
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # Same CPU partition as the ChimeraBoost / CTBoost runs (16 vCPUs, 64 GB RAM) for comparable
    # timings; 200 array tasks at once instead of the default 100.
    scheduler_setup=GCPSlurmSetup(
        bundle_size=2,
        cpu_partition="cpun416mtspotinteractive",
        array_job_limit=200,
    ),
)
plan.setup_jobs()
```

---

## 2026-09-09 — ctboost_09092026

- **Model(s):** CTBoost (all configs)
- **Git SHA:** `682365c1`
- **Purpose:** First full TabArena-v0.1 run of CTBoost (conditional-inference-tree gradient
  booster, https://github.com/captnmarkus/ctboost) for its integration in
  https://github.com/autogluon/tabarena/pull/479, pinned to `ctboost==0.1.61`. Processed and
  uploaded as suite `tabarena-2026-09-09` (`ctboost_method_metadata`), registered in the arena
  collection as a verified model.
- **Notes:** Same shape as the CPU tree boosters (ChimeraBoost, Perpetual): full task set (all
  splits), default config + the full 200-config HPO space, CPU partition `cpun416mtspotinteractive`
  (16 vCPUs, 64 GB RAM), `memory_limit`/`num_cpus` left `None` so node values are picked up, bundle
  size 2. Extra dep in the run venv: `ctboost==0.1.61`. CPU model, so no `fake_memory_for_estimates`;
  `CTBoostModel._estimate_memory_usage_static` is compared against node RAM to budget the parallel
  bagging folds. No untimed warm-up hook (`CTBoostModel` has no `warmup` classmethod), so any
  first-call cost of the library lands inside the first timed fit of each worker. The wrapper caps
  the native histogram threads through `CTBOOST_HIST_THREADS`, which the `.so` reads per fit, so the
  cap holds in Ray fold workers. Supersedes the lite / 25-config run `ctboost_08092026` on 0.1.60
  (2026-09-08, job 1100344), which was never uploaded. About 26 h of cluster time (2026-09-09 13:42
  to 2026-09-10 16:01); 164,016 result files (816 splits x 201 configs).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="ctboost_09092026",
    model_jobs=[
        ModelJob(models=("CTBoost", "all"), name="cpu"),
    ],
    task_subset=TaskSubset(),  # all splits of every task
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # Same CPU partition as the ChimeraBoost runs (16 vCPUs, 64 GB RAM) for comparable timings.
    scheduler_setup=GCPSlurmSetup(bundle_size=2, cpu_partition="cpun416mtspotinteractive"),
)
plan.setup_jobs()
```

---

## 2026-08-10 — chimeraboost_10082026

- **Model(s):** ChimeraBoost (all configs)
- **Git SHA:** `c305ab69`
- **Purpose:** Full rerun on ChimeraBoost 0.30.0 (requested in
  https://github.com/autogluon/tabarena/issues/463), which reworks the algorithm for better
  regression and small-data accuracy and better speed on large data. The registered baseline
  (suite `tabarena-2026-07-13`) is 0.14.1, so this repeats that run's shape to keep accuracy and
  timings comparable. Processed and uploaded as suite `tabarena-2026-08-10`
  (`chimeraboost_v030_method_metadata`), which replaces 0.14.1 in the arena collection.
- **Notes:** Same shape as both 0.14.1 runs: full task set (all splits), default config + the full
  200-config HPO space, CPU partition `cpun416mtspotinteractive` (16 vCPUs, 64 GB RAM, 0 GB VRAM),
  `memory_limit`/`num_cpus` left `None` so node values are picked up, bundle size 10. Extra dep in
  the run venv: `chimeraboost>=0.30.0`. Warm-up stays untimed (`ChimeraBoostModel.warmup`
  pre-compiles the numba kernels outside the fit and numba's disk cache carries them into the fold
  workers). 0.30.0's `refit_full="replay"` default is inert by design here: it only fires for fits
  that use ChimeraBoost's own internal split, and the wrapper passes AutoGluon's bagging validation
  fold as an explicit `eval_set` — refitting on that fold would train on the rows whose predictions
  become the out-of-fold predictions used for scoring and ensembling. About 10 h of cluster time
  (09:34 to 19:45).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="chimeraboost_10082026",
    model_jobs=[
        ModelJob(models=("ChimeraBoost", "all"), name="cpu"),
    ],
    task_subset=TaskSubset(),  # all splits of every task, as in the 0.14.1 runs
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # Same CPU partition as the 0.14.1 runs (16 vCPUs, 64 GB RAM) for comparable timings.
    scheduler_setup=GCPSlurmSetup(bundle_size=10, cpu_partition="cpun416mtspotinteractive"),
)
plan.setup_jobs()
```

---

## 2026-07-10 — tabdptturbo_10072026

- **Model(s):** TabDPT-Turbo (0 — single default config, no HPO)
- **Git SHA:** `bb2df761`
- **Purpose:** First TabArena-v0.1 benchmark of TabDPT-Turbo (TabDPT v1.2), added as a
  `TabDPTTurboModel` subclass alongside the v1.1 `TabDPTModel` (shared `TabDPTModelBase`).
  Foundation model (in-context learning), single-node GCP GPU run. Supports binary / multiclass /
  regression, so all problem types are included.
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU, `bundle_size=1`, full task set
  (`TaskSubset()` — all splits). `NUM_CONFIGS=0`: one in-context-learning default config per task,
  no HPO search space (`can_hpo=False`). No `fake_memory_for_estimates`: the wrapper cannot estimate
  memory statically (`can_estimate_memory_usage_static=False`), so there is no VRAM estimate to
  correct. Extra dep in the run venv: `tabdpt>=1.2.0` (this also upgrades the shared v1.1 install;
  each wrapper pins its own checkpoint via `model_weight_path`, so v1.1 still loads
  `tabdpt1_1.safetensors` and Turbo loads `tabdpt1_2.safetensors` from `Layer6/TabDPT`). Turbo
  defaults: `context_reduction="subsample"`, `clip_sigma=8`, `compile=False` (torch.compile is off
  for pickling/refit robustness across many small bagged fits — the speedup comes from subsample
  context + the v1.2 weights). Eval scored on `[[], ["binary"], ["multiclass"], ["regression"]]`.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="tabdptturbo_10072026",
    model_jobs=[
        ModelJob(
            models=("TabDPT-Turbo", 0),
            name="gpu",
            resources={"num_gpus": 1},
        ),
    ],
    context=TabArenaContext(),
    task_subset=TaskSubset(),  # full task set (all splits)
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000spotinteractive", bundle_size=1),
)
benchmark_plan.setup_jobs()
```

---

## 2026-07-13 — chimeraboost_13072026

- **Model(s):** ChimeraBoost (all configs)
- **Git SHA:** `99d494d3` (+ the `add_chimera_fast` metadata split: rerun recorded under
  `chimeraboost_new_method_metadata`, suite `tabarena-2026-07-13`)
- **Purpose:** Rerun of `benchmark_chimeraboost_16062026` with the untimed environment warm-up
  API (#441): the original run's fit times include ChimeraBoost's numba JIT compilation (~10s per
  cold environment). `ChimeraBoostModel.warmup` now pre-compiles the kernels untimed (warm-up is
  on by default in the experiment runner) and numba's disk cache carries them into fold workers.
- **Notes:** Mirrors the original run's shape for comparable timings: full task set (all splits),
  default config + full 200-config HPO space, CPU partition `cpun416mtspotinteractive`
  (16 vCPUs, 64 GB RAM, 0 GB VRAM), `memory_limit`/`num_cpus` left `None` so node values are
  picked up, bundle size 10. Extra dep in the run venv: `chimeraboost>=0.14.1` (adds
  `chimeraboost.warmup()`). New `(method, suite)` key keeps results separate from the superseded
  suite `tabarena-2026-06-30`, which stays registered in the arena collection until the rerun is
  processed/uploaded (the upload step swaps the registration).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaBenchmarkPlan(
    benchmark_name="chimeraboost_13072026",
    model_jobs=[
        ModelJob(models=("ChimeraBoost", "all"), name="cpu"),
    ],
    context=TabArenaContext(),
    task_subset=TaskSubset(),  # all splits of every task (full rerun, like the original)
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # Same CPU partition as the original run (16 vCPUs, 64 GB RAM) for comparable timings.
    scheduler_setup=GCPSlurmSetup(bundle_size=10, cpu_partition="cpun416mtspotinteractive"),
)
plan.setup_jobs()
```

---

## 2026-07-07 — tabfmplus_07072026

- **Model(s):** TabFM+ (system — single default config, 0 random)
- **Git SHA:** `f6815255`
- **Purpose:** TabFM+ — TabFM run through its heavier `ensemble` interface, wrapped as an
  `ExternalSystemModel` (`TabFMPlusSystemModel`) and benchmarked *as a system* (not an
  AutoGluon registry model) — on TabArena-v0.1, single-node GCP GPU run. Supports binary /
  multiclass / regression, so all problem types are included.
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU, `bundle_size=1`,
  `time_limit=4h`. **System run**, not a model name: `experiment_bundle` is built with
  `system_experiments=True` and the job entry is the `gen_tabfm_plus` `SystemConfigGenerator`
  (a `(generator, n_configs)` tuple). That gives one direct fit on all training data — no
  bagging / weighted-ensemble — with the per-split seed threaded into the fit as `random_state`
  by the runner. `setup_ray_for_slurm_shared_resources_environment=False`: a single in-process
  system fit never uses Ray, and dropping the Ray setup removes one variable from the OpenBLAS
  thread-init deadlock hit on GPU nodes (the real fix is the BLAS-thread cap in
  `TabFMPlusSystemModel`; SHA `f6815255` = "fix for deadlock"; this is defense-in-depth).
  `prefetch_model_weights=False`: a system has no registry name to key weight prefetch off, and
  TabFM+ reuses TabFM's cached `google/tabfm-1.0.0-pytorch` HF checkpoint. Extra dep in the run
  venv: `tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@633cd265f498e1d20c9625be0639f6305d8e2541`.
  The `ensemble` interface (32 members + feature-cross / SVD schedules + CV calibration / NNLS
  blending) is much heavier than default TabFM — raise `time_limit` if large tasks time out.
  Eval matches the raw results by name prefix (system output lives under `TabFM+_c1_default`):
  `EvalMethod("TabFM+", ag_name_override="TabFM+", only_load_cache=True)` with `only_valid_tasks=True`
  so the leaderboard is scoped to the tasks TabFM+ actually ran. The script carries a `TEST_DATASET`
  single-dataset debug toggle (scoped to `Is-this-a-good-customer` at a short walltime to probe the
  GPU stall); it is `None` here → full TabArena-v0.1 suite.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabarena.systems.tabfm_plus import gen_tabfm_plus
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="tabfmplus_07072026",
    model_jobs=[
        ModelJob(
            # System entry: a `(SystemConfigGenerator, n_configs)` tuple (not a model name).
            models=(gen_tabfm_plus, 0),
            name="gpu",
            resources={"num_gpus": 1, "time_limit": 4 * 60 * 60},
        ),
    ],
    context=TabArenaContext(),
    task_subset=TaskSubset(dataset_names=None),  # full suite (TEST_DATASET debug toggle = None)
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2, system_experiments=True),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(
        gpu_partition="gpurtxpro6000spotinteractive",
        bundle_size=1,
        setup_ray_for_slurm_shared_resources_environment=False,
    ),
    prefetch_model_weights=False,
)

benchmark_plan.setup_jobs()
```

---

## 2026-07-07 — tabfm_07072026

- **Model(s):** TabFM (0 — default config only)
- **Git SHA:** `f6815255`
- **Purpose:** TabFM (in-context tabular foundation model, GPU) on TabArena-v0.1, single-node
  GCP GPU run over the full suite. Supports binary / multiclass / regression, so all problem
  types are included (unlike the regression-only Nori plan). Re-run of the earlier
  `tabfm_30062026` against the current build.
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU, `bundle_size=1`,
  `time_limit=8h`. Empty search space (`can_hpo=False`), so it runs its single default config
  (0 random configs). No `fake_memory_for_estimates`: TabFM reports
  `can_estimate_memory_usage_static = False` (no static estimate to compare against a budget),
  unlike the VRAM-faking DenseLight. Its `google/tabfm-1.0.0-pytorch` checkpoint is prefetched
  from Hugging Face by the registry's `prefetch_weights` before the parallel fits. Requires the
  `tabfm[pytorch]` dependency in the run venv. This snapshot runs the whole suite (`TaskSubset()`)
  at a flat 8h walltime, folding in the earlier run's per-dataset time-limit bumps
  (`customer_satisfaction_in_airline` / `APSFailure` / `GiveMeSomeCredit`).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="tabfm_07072026",
    model_jobs=[
        ModelJob(
            models=("TabFM", 0),
            name="gpu",
            resources={"num_gpus": 1, "time_limit": 8 * 60 * 60},
        ),
    ],
    context=TabArenaContext(),
    task_subset=TaskSubset(),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000spotinteractive", bundle_size=1),
)

benchmark_plan.setup_jobs()
```

---

## 2026-07-06 — tabswift_06072026

- **Model(s):** TabSwift (0 — default config only)
- **Git SHA:** `d2286d92`
- **Purpose:** First TabArena-v0.1 benchmark of TabSwift, a newly-integrated torch in-context-learning
  tabular foundation model (LAMDA-Tabular, ICML 2026), single-node GCP GPU run.
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU. **Full** benchmark —
  `task_subset=TaskSubset()` (every task + all splits/folds, not the `lite` first-split), all
  problem types (binary/multiclass/regression). Empty search space ⇒ single default config
  (0 random configs). TabSwift is **vendored** (`tabarena/models/tabswift/_vendor/`) so there is no
  pip extra to install; the `LAMDA-Tabular/TabSwift` `swift.ckpt` checkpoint is prefetched from
  Hugging Face by the registry before the fits. Wrapper uses `fold_fitting_strategy="sequential_local"`
  + `refit_folds=True` (shared HF cache; refit one model on all data at TFM parity). No
  `fake_memory_for_estimates` — TabSwift reports `can_estimate_memory_usage_static=False`;
  `num_cpus`/`memory_limit` left `None` so the node's values are picked up.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabarena.contexts import TabArenaContext
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaBenchmarkPlan(
    benchmark_name="tabswift_06072026",
    model_jobs=[
        ModelJob(
            models=("TabSwift", 0),
            name="gpu",
            resources={"num_gpus": 1},
        ),
    ],
    context=TabArenaContext(),
    task_subset=TaskSubset(),  # full benchmark — every task/split (not just "lite")
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000spotinteractive"),
)
plan.setup_jobs()
```

## 2026-06-18 — nori_regression_18062026

- **Model(s):** Nori (0 — default config only)
- **Git SHA:** `f2eca5c3`
- **Purpose:** Nori (regression-only GPU foundation model) on the TabArena-v0.1 regression subset, single-node GCP GPU run.
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU. Scoped to the `regression`
  task subset via `task_subset=TaskSubset(subset="regression")` — every regression task/split.
  Empty search space, so it runs its single default config (0 random configs).
  `num_cpus`/`memory_limit` left `None` so the node's values are picked up.

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

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="nori_regression_18062026",
    model_jobs=[
        # Nori is a regression-only GPU foundation model with an empty search space,
        # so it runs its single default config (0 random configs) on a GPU node.
        ModelJob(models=("Nori", 0), name="gpu", resources={"num_gpus": 1}),
    ],
    # The TabArena-v0.1 context owns the task metadata + subset predicates; `task_subset`
    # scopes `context.build_jobs`. `subset="regression"` keeps every regression task/split.
    context=TabArenaContext(),
    task_subset=TaskSubset(subset="regression"),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_18062026/bin/python",
    ),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000spotinteractive"),
)

benchmark_plan.setup_jobs()
```

---

## 2026-06-16 — benchmark_chimeraboost_16062026

- **Model(s):** ChimeraBoost (all configs)
- **Git SHA:** `68c1919d`
- **Purpose:** ChimeraBoost on TabArena-v0.1 tasks, single-node GCP CPU run.
- **Notes:** CPU partition `cpun416mtspotinteractive` (16 vCPUs, 64 GB RAM, 0 GB VRAM);
  `memory_limit`/`num_cpus` left `None` so the node's values are picked up. Bundle size 10.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    TabArenaBenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

benchmark_plan = TabArenaBenchmarkPlan(
    benchmark_name="benchmark_chimeraboost_16062026",
    model_jobs=[
        ModelJob(models=("ChimeraBoost", "all")),
    ],
    tasks=TaskMetadataCollection.from_preset("TabArena-v0.1"),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/beyondarena_27052026/bin/python",
    ),
    # Run on GCP
    # -> None for these two values so node values are picked up
    # -> CPU partition: 16 vCPUs, 64 GB RAM, 0 GB VRAM
    resources_setup=TabArenaV0pt1ResourcesSetup(memory_limit=None, num_cpus=None),
    scheduler_setup=GCPSlurmSetup(bundle_size=10, cpu_partition="cpun416mtspotinteractive"),
)

benchmark_plan.setup_jobs()
```

---

## 2026-05-22 — benchmark_tabpfn_wide_22052026

- **Model(s):** TabPFN-Wide (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup2026` API)_
- **Purpose:** TabPFN-Wide on TabArena-v0.1 tasks, single-node GCP.
- **Notes:** Constrained to ≤10k train samples / ≤10 classes, classification only.
  GPU partition `gpua100highmemoryspotmt`, exclusive node, array job limit 100.

```python
from tabflow_slurm.setup_slurm_base_v2 import BenchmarkSetup2026, PathSetup, SlurmSetup

@dataclass
class ExtraPathSetup(PathSetup):
    base_path: str = "/path/to/workspace/"
    tabarena_repo_name: str = "XXX"
    venv_name: str = "XXXX"
    openml_cache_from_base_path: str | Literal["auto"] = "auto"


@dataclass
class TabArenaV0pt1SingleNodeBenchmarkSetup(BenchmarkSetup2026):
    shuffle_features: bool = False
    n_random_configs: int = 200
    dynamic_tabarena_validation_protocol: bool = False
    preprocessing_pipelines: list[str] = field(default_factory=lambda: ["default"])
    memory_limit: None = None
    num_cpus: None = None


TabArenaV0pt1SingleNodeBenchmarkSetup(
    benchmark_name="benchmark_tabpfn_wide_22052026",
    task_metadata="tabarena-v0.1",
    num_gpus=1,
    models=[
        ("TabPFN-Wide", 0),
    ],
    custom_model_constraints={
        "TA-TABPFN-WIDE": {
            "max_n_samples_train_per_fold": 10_000,
            "max_n_classes": 10,
            "regression_support": False,
        },
    },
    path_setup=ExtraPathSetup(),
    slurm_setup=SlurmSetup(
        gpu_partition="gpua100highmemoryspotmt",
        cpu_partition="cpuhighmem16mtspot",
        extra_gres=None,
        exclusive_node=True,
    ),
).setup_jobs(array_job_limit=100)
```

---

## 2026-05-14 — benchmark_iltm_14052026

- **Model(s):** iLTM (25 configs)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** iLTM foundation model.
- **Notes:** Only 25 configs due to compute constraints, similar treatment to TabSTAR. 1 GPU, 1h limit.

```python
from tabflow_slurm.setup_slurm_base import BenchmarkSetup

BenchmarkSetup(
    benchmark_name="benchmark_iltm_14052026",
    models=[
        ("iLTM", 25),
    ],
    num_gpus=1,
    configs_per_job=1,
    time_limit=60 * 60 * 1,
).setup_jobs()
```

---

## 2026-05-14 — benchmark_orionmsp_14052026

- **Model(s):** OrionMSP-1.5 (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** OrionMSP foundation model.
- **Notes:** H200 partition, 140 GB fake VRAM for estimates, 2h limit. Classification
  only (like TabICLv1).

```python
BenchmarkSetup(
    benchmark_name="benchmark_orionmsp_14052026",
    models=[
        ("OrionMSP", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    time_limit=60 * 60 * 2,
    problem_types_to_run=["binary", "multiclass"],
).setup_jobs()
```

---

## 2026-05-11 — benchmark_tabpfn_3_11052026

- **Model(s):** TabPFN-3 (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** TabPFN-3 foundation model.
- **Notes:** H200 partition, 140 GB fake VRAM, 2h limit.

```python
BenchmarkSetup(
    benchmark_name="benchmark_tabpfn_3_11052026",
    models=[
        ("TabPFN-3", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    time_limit=60 * 60 * 2,
).setup_jobs()
```

---

## 2026-05-11 — limix_11052026

- **Model(s):** LimiX (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** LimiX foundation model.
- **Notes:** H200 partition, 140 GB fake VRAM, **4h limit** — predict on large data is very slow.

```python
BenchmarkSetup(
    benchmark_name="limix_11052026",
    models=[
        ("LimiX", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    time_limit=60 * 60 * 4,
).setup_jobs()
```

---

## 2026-03-25 — 250326_tabpfnv26

- **Model(s):** TabPFN-2.6 (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** TabPFN-v2.6 foundation model.
- **Notes:** H200 partition, 140 GB fake VRAM, 2h limit.

```python
BenchmarkSetup(
    benchmark_name="250326_tabpfnv26",
    models=[
        ("TabPFN-2.6", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    time_limit=60 * 60 * 2,
).setup_jobs()
```

---

## 2026-02-25 — perpetual_booster_25022026

- **Model(s):** PerpetualBooster (5 configs)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** Perpetual gradient booster.
- **Notes:** CPU run. 4h limit initially — can take a while to run.

```python
BenchmarkSetup(
    benchmark_name="perpetual_booster_25022026",
    models=[
        ("PerpetualBooster", 5),
    ],
    configs_per_job=1,
    time_limit=60 * 60 * 4,
).setup_jobs()
```

---

## 2026-02-14 — tabpicl_v2_14022026

- **Model(s):** TabICLv2 (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** TabICLv2 foundation model.
- **Notes:** H200 partition, 140 GB fake VRAM (so TabICL sees 140 GB VRAM).

```python
BenchmarkSetup(
    benchmark_name="tabpicl_v2_14022026",
    models=[
        ("TabICLv2", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
).setup_jobs()
```

---

## 2026-01-31 — tabstar_31012026

- **Model(s):** TabSTAR (25 configs)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** TabSTAR foundation model.
- **Notes:** 5 configs per job, model-agnostic preprocessing disabled.

```python
BenchmarkSetup(
    benchmark_name="tabstar_31012026",
    models=[
        ("TabSTAR", 25),
    ],
    num_gpus=1,
    configs_per_job=5,
    model_agnostic_preprocessing=False,
).setup_jobs()
```

---

## 2025-12-19 — ag_experiment_191225

- **Model(s):** AutoGluon `extreme_v150_4h`, AutoGluon `extreme_noncommercial_v150_4h`
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** AutoGluon new-presets benchmark (4h presets).
- **Notes:** `tabarena_lite=True`, 4h time limit, presets loaded from S3 URLs.

```python
BenchmarkSetup(
    benchmark_name="ag_experiment_191225",
    models=[
        (
            "AutoGluon_extreme_v150_4h",
            dict(
                fit_kwargs=dict(
                    presets="https://ag-presets.s3.us-west-2.amazonaws.com/presets/extreme_v150.yaml",
                ),
            ),
        ),
        (
            "AutoGluon_extreme_noncommercial_v150_4h",
            dict(
                fit_kwargs=dict(
                    presets="https://ag-presets.s3.us-west-2.amazonaws.com/presets/extreme_noncommercial_v150.yaml",
                ),
            ),
        ),
    ],
    num_gpus=1,
    time_limit=14400,
    configs_per_job=1,
    tabarena_lite=True,
).setup_jobs()
```

---

## 2025-11-24 — sap_rpt_oss_new_2411

- **Model(s):** SAP-RPT-OSS (0 — default config only)
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** ConTexTab / SAP RPT OSS benchmark.
- **Notes:** H200 partition for a large dataset (230 features, 33k train samples),
  140 GB fake VRAM, model-agnostic preprocessing disabled, 5h limit.

```python
BenchmarkSetup(
    benchmark_name="sap_rpt_oss_new_2411",
    models=[
        ("SAP-RPT-OSS", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    model_agnostic_preprocessing=False,
    time_limit=5 * 60 * 60,
).setup_jobs()
```

---

## 2025-11-14 — tabpfnv25_hpo_14112025

- **Model(s):** RealTabPFN-v2.5 — two complementary runs: all configs on small data, 25 configs on large data
- **Git SHA:** _pre-refactor (`BenchmarkSetup` API)_
- **Purpose:** TabPFN-v2.5 with HPO search space.
- **Notes:** Split into two jobs sharing one `benchmark_name`. **Run 1** (small data,
  ≤50k train): all configs, 10 per job. **Run 2** (large data, 50k–100k train): only
  25 configs (GPU memory + runtime limits, scarce large GPUs), H200 partition, 140 GB
  fake VRAM, `parallel_benchmark_fix="_large_vram"` so job scripts don't crash.

```python
# Run 1 — small data (<= 50k train), all configs
BenchmarkSetup(
    benchmark_name="tabpfnv25_hpo_14112025",
    models=[
        ("RealTabPFN-v2.5", "all"),
    ],
    num_gpus=1,
    configs_per_job=10,
    custom_model_constraints={
        "REALTABPFN-V2.5": {
            "max_n_samples_train_per_fold": 50_000,
            "max_n_features": 2000,
            "max_n_classes": 10,
        }
    },
).setup_jobs()

# Run 2 — large data (50k-100k train), 25 configs, big GPUs
BenchmarkSetup(
    benchmark_name="tabpfnv25_hpo_14112025",
    models=[
        ("RealTabPFN-v2.5", 25),
    ],
    num_gpus=1,
    configs_per_job=1,
    custom_model_constraints={
        "REALTABPFN-V2.5": {
            "max_n_samples_train_per_fold": 100_000,
            "max_n_features": 2000,
            "max_n_classes": 10,
            "min_n_samples_train_per_fold": 50_001,
        }
    },
    slurm_gpu_partition="alldlc2_gpu-h200",
    fake_memory_for_estimates=140,
    parallel_benchmark_fix="_large_vram",
).setup_jobs()
```
