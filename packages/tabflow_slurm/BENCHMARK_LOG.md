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
- **Validation protocol:** <the context's key, e.g. `8x1`; name any `official_validation_protocol=False` run>
- **Purpose:** <one line>
- **Notes:** <partition, constraints, runtime, anything non-obvious>

​```python
<verbatim setup_jobs() call as run>
​```
```

---

## 2026-09-21 — tabdpt13_21092026

- **Model(s):** TabDPT-1.3 (default config only, `NUM_CONFIGS=0`, all 816 splits)
- **Git SHA:** `003f2a4d` (branch `add-tabdpt-1.3`, PR #576, rebased on main `83036084`); editable AutoGluon
  `../autogluon` at `eeab9f95`
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** First TabArena-v0.1 benchmark of TabDPT v1.3 (Layer6, `TabDPTv13Model`, a subclass of
  `TabDPTTurboModel` that pins `tabdpt1_3.safetensors` at `Layer6/TabDPT` revision `a5ca6e01`), the release that
  replaces TabDPT-Turbo as the installable TabDPT entry.
- **Notes:** SkyPilot job pool `tabarena-tabdpt13-21092026` (`--scheduler skypilot-pool`), 8 spot `g4-standard-48`
  workers (RTX PRO 6000, 96 GB; `fake_memory_for_estimates=96`), bundle size 1, shared API server, bucket prefix
  `gs://p2or-sky-cache-eu-dev/lennart_priorlabs_ai/tabarena`, env manifest `d80639a72f41`, run venv
  `~/.venvs/tabarena_10082026` (Python 3.12, torch 2.13.0+cu130). Extra dep: `tabdpt @ git+https://github.com/
  layer6ai-labs/TabDPT-inference.git@336ed08f`, the 1.3.0 release plus the separable `TabDPTEstimator._load_model`
  (layer6ai-labs/TabDPT-inference#79, merged 2026-09-21, no release yet) that the wrapper declares as its
  shared-weights loader; the install replaces tabdpt 1.2.0 (TabDPT-Turbo, superseded). The checkpoint (252 MB, public)
  was seeded into the bucket cache and loaded offline. Default 1 h time limit per config,
  `context_reduction="subsample"`, `clip_sigma=8`, `compile=False` (as Turbo). Launch `tabdpt13_21092026_gpu-20260921-154715-c613` (816 bundles,
  managed jobs 4800-4807) submitted 15:53 UTC once the first workers were READY; the 816 items ran from 15:54 to
  16:22 UTC (28 min on 8 workers), all eight jobs SUCCEEDED, 0 failed items, 0 recoveries; pool taken down 16:25 UTC.
  Warm-up audit `ok` for all 816 items (mean warm-up 1.8 s), no cold imports and no CUDA initialisation inside the
  timed fit or predict. The leaderboard labels the run by its config type (`TA-TABDPT-1.3 (default)`) until the
  method is hosted.
  Result (full task set, 97 entrants): TabDPT-1.3 #18, Elo 1517 (+55/-39), normalized score 0.409, improvability
  14.2%, between RealTabPFN-v2.5 (tuned, 1526) and RealTabPFN-v2.5 (default, 1499); subsets: binary #17/95 (1526,
  +71/-57), multiclass #21/95 (1505, +199/-107), regression #21/94 (1643, +192/-121); median time per 1K rows: train
  0.57 s, inference 0.176 s. TabDPT-Turbo (v1.2) sits at #22 (Elo 1431; 2.07 s / 0.182 s per 1K) and TabDPT v1.1
  tuned + ensembled at #23 (1430); in regression alone v1.1 tuned + ensembled (1706) and tuned (1655) stay ahead of
  1.3 (1643).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="tabdpt13_21092026",
    model_jobs=[
        ModelJob(
            models=("TabDPT-1.3", 0),
            name="gpu",
            resources={"num_gpus": 1, "fake_memory_for_estimates": 96},
        ),
    ],
    task_subset=TaskSubset(),  # the full task set, all splits
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=SkyPilotSetup(
        bundle_size=1,
        workers=8,
        use_pool=True,
        pool_name="tabarena-tabdpt13-21092026",
    ),
)
plan.setup_jobs()
```

---

## 2026-09-16 — rerun_ebm_catboost_16092026

- **Model(s):** ExplainableBM (EBM) and CatBoost, default plus 200 random configs (`NUM_CONFIGS="all"`, 201 configs
  per split, 164,016 items per model, 328,032 in total) on all 816 splits
- **Git SHA:** `f87f22d3` (main) plus the fixes of PR #584 on `benchmark/rerun-new-pipeline-16092026` (merged as
  `e53cc52c`); editable AutoGluon `../autogluon` master `957883c9`
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** Re-run the two hosted CPU tree boosters under the timing and warm-up pipeline (PR stack #539 to #546,
  AutoGluon #5919 to #5921) so their hosted train and inference times come from the same measurement as the
  foundation models re-run in `rerun_tfms_16092026`.
- **Notes:** Hardware `n4-standard-16` spot (16 vCPU, 64 GB) throughout, as SLURM partition `cpun416mtspotinteractive`
  and as SkyPilot pool workers (`cpus=16, memory=64, use_spot`; the shared API server's admin policy rejects a pinned
  instance type and spread the pool over `europe-west4` and `asia-northeast1`). `num_cpus=None` resolves to the node's
  16 vCPUs. The run alternated between the two schedulers as spot capacity moved: SLURM array 1156150 (bundle 12,
  stopped by the maintainer, slow), SkyPilot pools `tabarena-rerun-16092026-cpu` and `-cpu2` (600 workers each; both
  died of control-plane overload: SQLite locks and zombie `CANCELLING` jobs on the jobs controller, replica-manager
  assertion errors on the pool controller; the VMs had to be deleted with `gcloud`), pool `-cpu3` (250 workers, about
  160 to 215 alive under spot reclaims; CatBoost launch `cpu-20260917-073825-896d`, remainder
  `cpu_catboost_rest-20260918-004301-ef34`, EBM tail `cpu_ebm_tail_sky-20260918-033957-17c3`) and SLURM arrays
  1159855, 1162259, 1162810 (EBM, superseded by a clean restart), 1166154 (EBM, all remaining items, bundle 12) and
  1169391 (EBM, the last 2,808 items, bundle 4). Every hand-over between schedulers was a cache-aware `setup` after
  cancelling the other side, so no item was fitted twice on purpose (a handful straddling a dataset boundary may
  have been). Infrastructure lessons went into the `benchmark-model` skill and PR #584: one array per run,
  `scontrol release` for tasks parked as `launch failed requeued held`, no pulls into a checkout that SLURM imports
  live, a SLURM `setup` does not sync the bucket (sync before mixing schedulers), stuck `RECOVERING` pool jobs hold
  their claims until cancelled, a worker's per-item scratch must stay under a short path (Unix socket limit hit by
  EBM's multiprocessing and Ray). Spot preemption was heavy (about 510 SLURM requeues); 5 items failed for reasons
  other than preemption (a registry import during a branch update) and were refit by the follow-up setup. Result:
  CatBoost 164,016 results, no failed item; processed and uploaded as suite `tabarena-2026-09-16` in PR #584 with
  Elo identical to the July suite (default 1378, tuned 1407, tuned + ensembled 1419), but the median train time per
  1K rows rose from 5.88 to 8.68 s (default) and 1346 to 2081 s (tuned). NOT HOSTED in the end: the maintainer
  reverted the collection to the July suite `tabarena-2026-07-13` and the r2 artifacts were deleted again (follow-up
  PR), because the two runs' fit times are not comparable until it is settled which CPU count the tree boosters
  should be timed with (the 16 vCPUs of an n4-standard-16 are 8 physical cores) and on which hardware the July
  suite ran. EBM: 164,016 results, no failed item, finished 2026-09-18 17:20 UTC; processed with 17 `Not close TEST` warnings (superconductivity and physiochemical_protein, 0.007 to 0.028 percent of the test rows of one bagged config each, float rounding in the re-aggregated bag predictions). Joint leaderboard of the run's eval: default Elo 1188, tuned 1219, tuned + ensembled 1256; median train time per 1K rows 10.70 s (default) and 2674 s (tuned) against 6.67 s and 1711 s in the July suite, the same 1.5 to 1.6 times slowdown as CatBoost; median predict time unchanged. Also not hosted, for the same reason; its upload was stopped and removed. The raw results of both models stay in the workspace (`output/rerun_ebm_catboost_16092026/data`) for the timing investigation. About 3,900 node-hours of n4-standard-16 across both models.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    GCPSlurmSetup,
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

MODELS = ("ExplainableBM", "CatBoost")
plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="rerun_ebm_catboost_16092026",
    model_jobs=[
        ModelJob(models=[(model, "all") for model in MODELS], name="cpu"),
    ],
    task_subset=TaskSubset(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_rerun_16092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # SLURM launches (the EBM arrays; bundle_size=4 for the final 2,808 heavy items):
    scheduler_setup=GCPSlurmSetup(cpu_partition="cpun416mtspotinteractive", bundle_size=12, array_job_limit=300),
    # SkyPilot launches (CatBoost and the EBM tail) used instead:
    # scheduler_setup=SkyPilotSetup(bundle_size=20, workers=250, cpu_cpus="16", cpu_memory="64", use_pool=True,
    #                               pool_name="tabarena-rerun-16092026-cpu3", api_server_endpoint="http://skypilot-api:46580"),
)
plan.setup_jobs()
# Relaunches under the same benchmark name scoped with ModelJob(models=[(model, "all")], tasks=TaskSubset(dataset_names=[...]))
# when a dataset group moved between the schedulers.
```

---

## 2026-09-17 — rerun_tabpfn35_17092026

- **Model(s):** TabPFN-3.5 and TabPFN-3.5-Fast (default config only, `NUM_CONFIGS=0`, all 816 splits each)
- **Git SHA:** `4f6dc780` (branch `benchmark/rerun-new-pipeline-16092026`, PR #584), editable AutoGluon
  `../autogluon` master `957883c9`
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** Re-time the fits of run `tabpfn35_17092026` (suite `tabarena-2026-09-17`, entry below) after dropping
  `n_preprocessing_jobs=num_cpus` from the wrapper. Starting tabpfn's preprocessing worker pool cost a fixed 15 to
  20 s per fit, so the hosted fit times carried that constant (no TabPFN-3.5 fit under 16.0 s per split, no Fast
  fit under 20.2 s, against 2.2 s for TabPFN-3 whose wrapper passes `n_jobs`, which tabpfn ignores). Inference
  and the predictions were never affected.
- **Notes:** SkyPilot job pool `tabarena-tabpfn35-fix-17092026` (`--scheduler skypilot-pool`), 32 spot
  `g4-standard-48` workers (RTX PRO 6000, 96 GB; `fake_memory_for_estimates=96`), bundle size 1, both models in one
  `ModelJob` (1632 items), shared API server `http://skypilot-api:46580`, run venv
  `~/.venvs/tabarena_tabpfn35_17092026` (Python 3.12, torch 2.14.0+cu130, tabpfn 9.0.0, editable AutoGluon).
  Launch `rerun_tabpfn35_17092026_gpu-20260917-203001-791c` (managed jobs 4218-4249); pool created 20:34 UTC, jobs
  launched 20:46 after a head-node restart, all 1632 items done by 21:14 (no failed item); pool taken down 21:20.
  Both checkpoints seeded from the head node's `~/.cache/tabpfn` (`offline_weights=True`). Per split against the
  flawed run: median fit 17.0 s to 2.2 s (TabPFN-3.5) and 21.2 s to 0.9 s (Fast), identical `metric_error` on every
  split (same seeds), inference within 0.01 s. Result (full task set, joint leaderboard with the hosted rows): Elo
  and normalized error identical to the hosted entries (TabPFN-3.5 1868, 0.161; Fast 1774, 0.252); median train
  time per 1K rows 5.67 to 1.84 s and 5.80 to 0.72 s, median predict time unchanged (0.47 s and 0.15 s). Processed
  and uploaded as suite `tabarena-2026-09-17-fix` (`tabpfn_3_5_method_metadata`, `tabpfn_3_5_fast_method_metadata`
  switched in place; r2 `cache/artifacts/tabarena-2026-09-17-fix/methods/<Method>/`); the `tabarena-2026-09-17`
  artifacts were deleted from r2.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

MODELS = ("TabPFN-3.5", "TabPFN-3.5-Fast")
plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="rerun_tabpfn35_17092026",
    model_jobs=[
        ModelJob(
            models=[(model, 0) for model in MODELS],
            name="gpu",
            resources={"num_gpus": 1, "fake_memory_for_estimates": 96},
        ),
    ],
    task_subset=TaskSubset(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_tabpfn35_17092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=SkyPilotSetup(
        bundle_size=1,
        workers=32,
        secrets=("HF_TOKEN",),
        use_pool=True,
        pool_name="tabarena-tabpfn35-fix-17092026",
        api_server_endpoint="http://skypilot-api:46580",
    ),
)
plan.setup_jobs()
```

---

## 2026-09-17 — limix2_17092026

- **Model(s):** LimiX-2 (default config only, `NUM_CONFIGS=0`, all 816 splits)
- **Git SHA:** `952dfdf0` (branch `limix_2`, PR #575, stacked on PR #584's `benchmark/rerun-new-pipeline-16092026` at
  `3040a65d`); editable AutoGluon `../autogluon` at `4db475bf` (master `957883c9` plus the `get_info` fix of
  autogluon/autogluon#5925, merged upstream 2026-09-17)
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** Maintainer re-run of the LimiX-2 submission (Stable AI, 400M-parameter in-context foundation model,
  PR #575, LimiX inference commit `774aa3e1`, checkpoint `stable-ai/LimiX-2` revision `20c07a07`) on the full TabArena task set.
- **Notes:** SkyPilot job pool `tabarena-limix2-17092026` (`--scheduler skypilot-pool`), 64 spot `g4-standard-48`
  workers (RTX PRO 6000, 96 GB; `fake_memory_for_estimates=96`), bundle size 1, shared API server
  `http://skypilot-api:46580`, bucket prefix `gs://p2or-sky-cache-eu-dev/lennart_priorlabs_ai/tabarena`, run venv
  `~/.venvs/tabarena_10082026` (Python 3.12, torch 2.13.0+cu130, `LimiX @ git+...@774aa3e1` installed with `--no-deps`
  plus `nvtx`, since the package pins torch 2.9.1). **Protocol deviation:** two `ModelJob`s with longer time limits than
  the 1 h default, `time_limit=2h` for 42 datasets (`gpu`, 735 items) and `time_limit=8h` for the nine large or wide
  datasets the authors named (`gpu_8h`, 81 items: APSFailure, Bioresponse, customer_satisfaction_in_airline,
  Diabetes130US, GiveMeSomeCredit, hiva_agnostic, kddcup09_appetency, QSAR-TID-11, SDSS17), following the TabFM
  precedent of `rerun_tfms_16092026`; an in-context model has no early stopping, so the budgets only decide whether an
  item completes. Three launches were needed. Launch 1 (13:25 UTC, env `54b9527da02c`, tabarena `441b9980`): every
  finished item failed in `post_evaluate` because the refit child's pickle raised `Can't get local object
  'RebalanceFeatureDistribution._set.<locals>.<lambda>'` (LimiX built its `power` preprocessing members from lambdas
  at their first transform, inside predict) and `BaggedEnsembleModel.get_info` summed the resulting `None` memory size
  (`TypeError`); fixed upstream in AutoGluon (autogluon/autogluon#5925) and in LimiX (`774aa3e1`, static methods with
  identical bodies, predictions unchanged). Launch 2 (14:50 UTC, env `92fb22b79e87`) ran on a rolled pool whose 30
  leftover version-1 workers still failed; the 183 items that finished on version-2 workers were kept. Launch 3
  (15:27 UTC, env `b136cf77d8a7`, pool recreated after `pool down`): `limix2_17092026_gpu-20260917-151555-b699`
  (552 bundles, managed jobs 4007-4070, done 16:45 UTC) and `limix2_17092026_gpu_8h-20260917-151805-d1cb` (81 bundles,
  jobs 3943-4006 plus 4217; the last item, an APSFailure split, finished 23:40 UTC; APSFailure takes about 3.25 h per split, so the whole 8h group ran about 8 h on the 64-worker pool). Two APSFailure claims were orphaned when
  SkyPilot "recovered" their jobs after controller status-check timeouts and the recovered worker did not recognise
  its own claims (fixed in #587); the claim objects were deleted and one more job launched. Warm-up audit `ok` for
  every item (mean warm-up 4.1 s), no cold imports or CUDA initialisation in the timed sections. Result (full task
  set, 96 entrants): LimiX-2 #1, Elo 1943 (+118/-80), normalized score 0.922, improvability 3.4%; #1 in the
  binary (1912, +142/-88), multiclass (1988, +396/-195) and regression (2200, +341/-175) subsets, ahead of TabPFN-3.5
  (1861) and TabFM+ (1821) overall; median time per 1K rows: train 30.94 s, inference 9.028 s
  (TabPFN-3.5: 5.7 s / 0.5 s). Its predecessor LimiX (v1) sits at #35 (Elo 1344).

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

MODEL = "LimiX-2"
LONG_JOB_DATASETS = (
    "APSFailure",
    "Bioresponse",
    "customer_satisfaction_in_airline",
    "Diabetes130US",
    "GiveMeSomeCredit",
    "hiva_agnostic",
    "kddcup09_appetency",
    "QSAR-TID-11",
    "SDSS17",
)
gpu_resources = {"num_gpus": 1, "fake_memory_for_estimates": 96}
plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="limix2_17092026",
    model_jobs=[
        ModelJob(
            models=[(MODEL, 0)],
            name="gpu",
            resources={**gpu_resources, "time_limit": 2 * 60 * 60},
            tasks=TaskSubset(dataset_names=default_budget_datasets),  # every TabArena dataset not in LONG_JOB_DATASETS
        ),
        ModelJob(
            models=[(MODEL, 0)],
            name="gpu_8h",
            resources={**gpu_resources, "time_limit": 8 * 60 * 60},
            tasks=TaskSubset(dataset_names=list(LONG_JOB_DATASETS)),
        ),
    ],
    task_subset=TaskSubset(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=SkyPilotSetup(
        bundle_size=1,
        workers=64,
        secrets=("HF_TOKEN",),
        use_pool=True,
        pool_name="tabarena-limix2-17092026",
        api_server_endpoint="http://skypilot-api:46580",
    ),
)
plan.setup_jobs()
```

---

## 2026-09-17 — tabpfn35_17092026

- **Model(s):** TabPFN-3.5 and TabPFN-3.5-Fast (default config only, `NUM_CONFIGS=0`, all 816 splits each)
- **Git SHA:** `33d2d4f4` (branch `add-tabpfn-3.5`, PR #577, stacked on PR #584's `benchmark/rerun-new-pipeline-16092026`
  at `c3182f62`; the same commits are `56c66736..b2cfc16c` after the rebase onto `f185c68a`), editable AutoGluon
  `../autogluon` master `957883c9`
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** First TabArena run of the September 2026 TabPFN release (tabpfn 9.0.0): the flagship TabPFN-3.5 and
  the smaller TabPFN-3.5-Fast, one multitask checkpoint each (`tabpfn-v3.5-20260909.safetensors`,
  `tabpfn-v3.5-fast-20260909.safetensors`, HF repo `Prior-Labs/tabpfn_3_5`).
- **Notes:** SkyPilot job pool `tabarena-tabpfn35-17092026` (`--scheduler skypilot-pool`), 32 spot `g4-standard-48`
  workers (RTX PRO 6000, 96 GB; `fake_memory_for_estimates=96`), bundle size 1, both models in one `ModelJob`
  (1632 items), shared API server `http://skypilot-api:46580`, bucket prefix
  `gs://p2or-sky-cache-eu-dev/lennart_priorlabs_ai/tabarena`, run venv `~/.venvs/tabarena_10082026` (Python 3.12,
  torch 2.13.0+cu130, tabpfn 9.0.0). Env manifest `env/48d8015a2f74`, launch `tabpfn35_17092026_gpu-20260917-073728-6334`
  (managed jobs 3323-3354). Pool applied 07:42 UTC, jobs launched 07:49, all 1632 bundles done 08:42 UTC (about
  52 min, no failed item, no relaunch, no imputation); pool taken down after the eval. Weights: tabpfn 9.0.0's own
  downloader needs a Prior Labs token (`TABPFN_TOKEN`) in a headless session, so both checkpoints were placed in
  `~/.cache/tabpfn` from the HF repo beforehand; the prefetchers returned them and `setup` seeded them
  (`offline_weights=True`). Two fixes landed on the branch during the run: `TABPFN_CHECKPOINT_SUFFIXES` gains
  `.safetensors` (without it the seeding classified the files as unstaged and would have re-downloaded), and the
  eval's `name_prefix_raw` now selects the exact method folder (`TA-TabPFN-3.5` is a prefix of
  `TA-TabPFN-3.5-Fast`, so the first eval collected both methods and refused). Warm-up audit `ok` for all 1632
  items (mean warm-up 5.8 s / 5.2 s), no cold imports or CUDA initialisation in the timed sections. Result (full
  task set, 95 entrants): TabPFN-3.5 #1, Elo 1864 (+94/-67), normalized error 0.158, ahead of TabFM+ (1821) and
  Causilo (1788); TabPFN-3.5-Fast #5, Elo 1778 (+84/-62), between AutoGluon 1.6 (noncommercial, 4h) and TabFM.
  Median time per 1K rows: train 5.67 s / 5.80 s, inference 0.48 s / 0.16 s. Subsets: TabPFN-3.5 #1 binary
  (1845), #3 multiclass (1877), #1 regression (2103); TabPFN-3.5-Fast #4 binary (1785), #5 multiclass (1810),
  #8 regression (1911). TabPFN-3 (rerun row) sits at #11 with Elo 1629. Processed and uploaded as suite
  `tabarena-2026-09-17` (`tabpfn_3_5_method_metadata`, `tabpfn_3_5_fast_method_metadata` in
  `models/tabpfn_3_5/info.py`; r2 `cache/artifacts/tabarena-2026-09-17/methods/<Method>/`), registered in the
  arena collection as verified (maintainer sign-off 2026-09-17). Superseded the same day: the wrapper passed
  `n_preprocessing_jobs=num_cpus`, and starting that worker pool added a fixed 15 to 20 s to every timed fit
  (no TabPFN-3.5 fit under 16.0 s per split, no Fast fit under 20.2 s, against 2.2 s for TabPFN-3). The
  results were re-run as `rerun_tabpfn35_17092026` (entry above) and the suite's r2 artifacts were deleted.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

MODELS = ("TabPFN-3.5", "TabPFN-3.5-Fast")
plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="tabpfn35_17092026",
    model_jobs=[
        ModelJob(
            models=[(model, 0) for model in MODELS],
            name="gpu",
            resources={"num_gpus": 1, "fake_memory_for_estimates": 96},
        ),
    ],
    task_subset=TaskSubset(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_10082026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=SkyPilotSetup(
        bundle_size=1,
        workers=32,
        secrets=("HF_TOKEN",),
        use_pool=True,
        pool_name="tabarena-tabpfn35-17092026",
        api_server_endpoint="http://skypilot-api:46580",
    ),
)
plan.setup_jobs()
```

---

## 2026-09-16 — rerun_tfms_16092026

- **Model(s):** Causilo, Mitra-v2, Xiaomi-TabLDM, TabFM, TabSwift, TabICLv2, EXAONE-Tabular, TabPFN-3 (default
  config only, all 816 splits each) and Nori-30M (default config, the 222 regression splits)
- **Git SHA:** `f87f22d3` (main) plus the fixes of the rerun PR on branch `benchmark/rerun-new-pipeline-16092026`
  (listed in the notes); editable AutoGluon `../autogluon` master `957883c9`
- **Validation protocol:** `8x1` (TabArena default, asserted by the context)
- **Purpose:** Re-run every hosted foundation model under the timing and warm-up pipeline (PR stack #539 to #546,
  AutoGluon #5919 to #5921) to refresh the hosted train and inference times; first campaign run on SkyPilot
  (`--scheduler skypilot-pool`, PR #573).
- **Notes:** SkyPilot job pool `tabarena-rerun-16092026`, 32 spot `g4-standard-48` workers (RTX PRO 6000, 96 GB;
  `fake_memory_for_estimates=96`), bundle size 1, shared API server `http://skypilot-api:46580`, bucket prefix
  `gs://p2or-sky-cache-eu-dev/lennart_priorlabs_ai/tabarena`, run venv `~/.venvs/tabarena_rerun_16092026`
  (Python 3.12, torch 2.13.0+cu130, tabpfn 8.4.0, synthefy-nori 0.14.0). Env manifest `env/1b7b185f5888`
  (main and Nori-30M launches), `env/ffbec55b27d0` (fix launches). Launches: `gpu-20260916-164300-11c9` (6528
  bundles, 32 jobs, 17:29 to 01:30 UTC), `gpu_regression-20260916-165215-4f05` (Nori-30M, 222 bundles, 8 jobs,
  done 18:21), `gpu_fix-20260916-175055-954e` (Causilo, 816 bundles, done 19:47), `gpu_fix_regression-20260916-175343-576b`
  (TabICLv2 regression, 222 bundles, done 18:46) and `gpu_tabfm_8h-20260917-014736-77f0` (the 27 TabFM splits
  of APSFailure, GiveMeSomeCredit and customer_satisfaction_in_airline that hit AutoGluon's fold budget under the
  one-hour limit; redone with `time_limit=8h` like the hosted run `tabfm_07072026`, a documented protocol
  deviation for this model). Infrastructure fixes made during the run (the rerun PR): `GcsStorage.upload_dir`
  follows symlinks (HF snapshots never reached the bucket), `seed_model_weights` re-seeds an empty remote
  manifest and mirrors the head node's cached weights (`collect_weight_paths`) into the scratch root before the
  prefetch, HF `trees/` listings are seeded (tabpfn-style `snapshot_download` of a pinned commit needs them
  offline; every Causilo item failed without them), the TabICLv2 prefetcher fetches the regressor checkpoint and
  returns paths, the TabPFN-3 prefetcher downloads the v3 checkpoints only and returns paths, the worker builds
  its venv with `--no-deps` (a re-resolved freeze failed on `mlxtend` vs `matplotlib`), the printed `pool apply`
  drops `--workers`, and the worker keeps its per-item scratch under a short `/tmp` path (Ray plasma and
  multiprocessing sockets exceed 107 bytes under the launch directory). Smoke fits of all nine models passed on
  the head-node CPU. Result: 6750 result files; warm-up audit `ok` for every item, no cold imports or CUDA
  initialisation in the timed sections. Rerun rows match the hosted Elo within noise (Causilo 1785, Mitra-v2
  1764, EXAONE-Tabular 1737, TabPFN-3 1625, Xiaomi-TabLDM 1581, TabICLv2 1558, TabSwift 1327, Nori-30M
  regression 1733) while median train time per 1K rows drops 2 to 4 times (Causilo 2.01 to 0.68 s, EXAONE
  5.79 to 2.00 s, TabPFN-3 3.66 to 1.29 s, Xiaomi-TabLDM 4.85 to 1.12 s, TabICLv2 2.05 to 0.75 s, TabSwift
  1.18 to 0.57 s, Mitra-v2 66.4 to 57.1 s, TabFM 38.8 to 16.1 s); TabFM 1777 (hosted 1779) after the 8 h relaunch
  of its 27 large-table splits. About 300 GPU hours; the run finished 2026-09-17 04:27 UTC. Processed and
  uploaded as suite `tabarena-2026-09-16` (`causilo_new_method_metadata`, `mitra_v2_new_method_metadata`,
  `tabldm_new_method_metadata`, `tabfm_2026_09_method_metadata`, `tabswift_2026_09_method_metadata`,
  `tabiclv2_2026_09_method_metadata`, `exaone_tabular_new_method_metadata`, `tabpfn_3_2026_09_method_metadata`,
  `nori30m_new_method_metadata`), which now hold the nine slots in the arena collection while the replaced
  entries moved to `methods_superseded`.

```python
from tabarena.benchmark.experiment import TabArenaV0pt1ExperimentBundle
from tabarena.benchmark.task.metadata import TaskSubset
from tabflow_slurm import (
    ModelJob,
    PathSetup,
    SkyPilotSetup,
    TabArenaV0pt1BenchmarkPlan,
    TabArenaV0pt1ResourcesSetup,
)

ALL_TASK_MODELS = ("Causilo", "Mitra-v2", "Xiaomi-TabLDM", "TabFM", "TabSwift", "TabICLv2", "EXAONE-Tabular", "TabPFN-3")
gpu_resources = {"num_gpus": 1, "fake_memory_for_estimates": 96}
plan = TabArenaV0pt1BenchmarkPlan(
    benchmark_name="rerun_tfms_16092026",
    model_jobs=[
        ModelJob(models=[(model, 0) for model in ALL_TASK_MODELS], name="gpu", resources=gpu_resources),
        ModelJob(
            models=[("Nori-30M", 0)],
            name="gpu_regression",
            resources=gpu_resources,
            tasks=TaskSubset(subset="regression"),
        ),
    ],
    task_subset=TaskSubset(),
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_rerun_16092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=SkyPilotSetup(
        bundle_size=1,
        workers=32,
        secrets=("HF_TOKEN",),
        use_pool=True,
        pool_name="tabarena-rerun-16092026",
        api_server_endpoint="http://skypilot-api:46580",
    ),
)
plan.setup_jobs()
# Relaunches under the same benchmark name: Causilo (all tasks) and TabICLv2 (regression tasks) after the
# seeding fixes, and TabFM with resources={"num_gpus": 1, "fake_memory_for_estimates": 96, "time_limit": 8 * 60 * 60}.
```

---

## 2026-09-13 — causilo_13092026

- **Model(s):** Causilo (default config only, `NUM_CONFIGS=0`)
- **Git SHA:** `d5f45677` (PR #536 head, branch `pr-536-add-causilo`)
- **Purpose:** Maintainer re-run of the Causilo 1.0.0 submission (Nums AI pretrained ICL foundation
  model, https://github.com/nums-ai/causilo, PR https://github.com/autogluon/tabarena/pull/536) on the
  full TabArena-v0.1 task set to verify the self-reported leaderboard numbers.
- **Notes:** GPU model on `gpurtxpro6000flex` (RTX PRO 6000, 96 GB) with
  `fake_memory_for_estimates=96`; the wrapper fits folds `sequential_local` with `refit_folds=True`, so
  folds never share the card. Fixed recipe `n_estimators=8`, `random_state=42`, no search space. Extra dep
  in the run venv (`~/.venvs/tabarena_mitra_v2_10092026`): `causilo==1.0.0`; torch 2.13.0+cu130 was
  already present. Weights pinned to HF commit `94f2bd91` and prefetched on the head node. Launched first
  as job 1153347 with `bundle_size=1` (816 tasks, `--time=2:00:00`); per-task Ray/venv setup (~20 s)
  dwarfed the second-long fits, so the maintainer cancelled its pending tasks after ~40 had started and
  the remaining 791 items were relaunched as job 1153391 with `bundle_size=10` (80 tasks,
  `--time=11:00:00`), cache-aware. Wall time 2026-09-13 11:15 to 11:54 UTC; 816 result files, no failed
  task, no imputation. Longest single fit about 130 s (GiveMeSomeCredit, customer_satisfaction_in_airline
  about 100 s). PyTorch allocator OOM warnings on the wide datasets (Bioresponse, hiva_agnostic,
  kddcup09_appetency, QSAR-TID-11) are the library's own chunk-shrinking retry loop; all of them recovered.
  Result: #3/88 overall (Elo 1794 +90/-58) behind the systems TabFM+ and AutoGluon 1.6 (noncommercial),
  #1/86 multiclass, #3/85 regression, #5/86 binary; matches the PR's self-reported Elo of 1792.9.
  Processed and uploaded as suite `tabarena-2026-09-13` (`causilo_method_metadata`), registered in the
  arena collection as a verified model.

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
    benchmark_name="causilo_13092026",
    model_jobs=[
        ModelJob(
            models=("Causilo", 0),
            name="gpu",
            resources={"num_gpus": 1, "fake_memory_for_estimates": 96},
        ),
    ],
    task_subset=TaskSubset(),  # the full task set, all splits
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_mitra_v2_10092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # First launch (job 1153347) used bundle_size=1; relaunched as job 1153391 with bundle_size=10.
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000flex", bundle_size=10),
)
plan.setup_jobs()
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

## 2026-09-12 — mitrav2_12092026

- **Model(s):** Mitra-v2 (0 — single default config, no HPO: the frozen fine-tuning recipe is the method)
- **Git SHA:** `c71945a3`
- **Purpose:** Second TabArena-v0.1 benchmark of Mitra-v2 on the revised wrapper (cheaper fine-tuning loop,
  16,384-row prediction chunks, and the fine-tuning context of the first bag child reused by the other
  children; commits `2d0b267a`, `4a2dc589`, `c71945a3` on PR #520). Same task set and resources as
  `mitrav2_10092026`, written to a fresh output folder so the two runs can be compared split by split.
- **Notes:** GPU partition `gpurtxpro6000flex` (RTX PRO 6000, 96 GB VRAM), 1 GPU, exclusive node,
  `fake_memory_for_estimates=96`, 1 h fit budget per config, full task set (816 splits). This time
  `bundle_size=5`: 164 array tasks (163 of five splits, one of a single split) with `--time=6:00:00`
  (5 x 1 h plus the 1 h overhead) instead of the first run's 816 single-split tasks at 2 h. The
  scheduler's large-dataset rule caught no TabArena-v0.1 dataset, so the bundles are contiguous runs of
  the same dataset. Flagged at launch: on the first run's code APSFailure took about 80 min per
  split (fit plus inference) and kddcup09_appetency about 73 min, so the all-same-dataset bundles
  20 (5 x APSFailure) and 101 (5 x kddcup09_appetency) would have exceeded 6 h. The revised wrapper
  halved both (APSFailure 41 min, kddcup09_appetency 43 min per split at most), so bundle 20 ran
  3 h 47 min and bundle 101, the longest array task, 4 h 04 min. Outcome: SLURM array 1150009
  (submitted 2026-09-12 17:55) completed 164/164 tasks with 816/816 results and no failures, first
  task start 18:00, last task end 05:40 the next day (about 11.7 h wall); 108 GPU-hours of measured
  fit plus inference against 139 for the first run. Each split is its own python process writing its
  own `results.pkl`, so a timed-out bundle would only have lost its remaining splits;
  `bundle_size_per_dataset` is the escape hatch for slower models. Launched from
  `tmp_scripts/run_mitra_v2.py setup` in the tabarena-edit-copy clone; venv
  `tabarena_mitra_v2_10092026` imports the `add-mitra-v2` checkout. Eval on
  `[[], ["binary"], ["multiclass"], ["regression"]]` (2026-09-13): Elo 1770 (position 3) on the full
  leaderboard against 1766 for the first run, normalized error 0.214 against 0.224; binary 1746,
  multiclass 1826, regression 1985. Mean train time 762 s and test inference 20 s per task against
  930 s and 92 s. Result processing printed one bag-consistency warning (hiva_agnostic split 1:
  stored bagged test predictions and the mean of the per-child predictions differ beyond rtol 5e-4
  on 64% of rows). Benign: the absolute gap is at most 7e-4 (mean 1e-4), the log loss agrees to four
  decimals, and the first run showed the same warning on five of the nine hiva_agnostic splits;
  near-zero probabilities fail a relative tolerance on GPU-level noise. The leaderboard uses the
  child-averaged predictions. Extra dep in the run venv: `autogluon.tabular[mitra]>=1.6,<1.7`;
  `flash-attn` not installed. Processed and uploaded 2026-09-13 as suite `tabarena-2026-09-12`
  (`mitra_v2_method_metadata`, r2://tabarena/cache/artifacts/tabarena-2026-09-12/methods/Mitra-v2) and
  registered in the arena collection next to Mitra v1 as a verified model.

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
    benchmark_name="mitrav2_12092026",
    model_jobs=[
        ModelJob(
            models=("Mitra-v2", 0),
            name="gpu",
            resources={
                "num_gpus": 1,
                # The gpu_partition's VRAM in GB (gpurtxpro6000flex -> RTX PRO 6000 -> 96).
                "fake_memory_for_estimates": 96,
            },
        ),
    ],
    task_subset=TaskSubset(),  # full task set (all splits)
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_mitra_v2_10092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000flex", bundle_size=5),
)
plan.setup_jobs()
```

---

## 2026-09-11 — xiaomitabldm_11092026

- **Model(s):** Xiaomi-TabLDM (default config only, no HPO configs)
- **Git SHA:** `fbfc63f0`
- **Purpose:** First full TabArena-v0.1 run of Xiaomi-TabLDM (tabular foundation model with a
  dual-stream column embedder and MoE backbone, https://huggingface.co/occams/Xiaomi-TabLDM) for its
  integration on branch `feature/add-TabLDM`, pinned to commit `6773a30d` of
  https://github.com/xiaomi-research/xiaomi-tabldm (not on PyPI, installed through the `tabldm` extra).
- **Notes:** Full task set (all splits), default config only: `gen_tabldm` has an empty search space,
  so the frozen in-context-learning recipe is the method (`n_estimators=8` forward passes per fold).
  GPU partition `gpurtxpro6000flex` (RTX PRO 6000, 96 GB), `fake_memory_for_estimates=96`,
  `memory_limit`/`num_cpus` left `None` so node values are picked up, bundle size 10 (82 array tasks,
  11 h wall each, 100 concurrent). A first bundle-size-1 array (1142547, 816 tasks) was cancelled
  minutes after launch at the maintainer's request before any result was written; the run is array
  1142639. Run venv `tabarena_mitra_v2_10092026` (this clone's venv). Both checkpoints
  (`clf_default.ckpt`, `reg_default.ckpt`) were prefetched on the head node by `setup`. The wrapper pins
  `fold_fitting_strategy="sequential_local"` and `refit_folds=True`, has no static memory estimate, and
  ignores `X_val`/`time_limit` (no training loop), so the 1 h per-config budget was never approached:
  the slowest split took 97 s for fit plus prediction and the longest task 30 min. All 816 splits
  completed without a failure. Wall span 16:52 to 19:41; the last hour was spent waiting for flex
  nodes to boot (partition-wide `NOT_RESPONDING+POWERING_UP` limbo affecting every user's new tasks):
  six tasks sat in CONFIGURING on nodes that never came up and were requeued by hand with
  `scontrol requeue`, one of them twice. The CPU smoke fit on the head node passed in 100 s.
  Eval: #8/88 overall (Elo 1588 +69/-61), #9/86 binary, #10/86 multiclass, #8/85 regression.
  Processed and uploaded 2026-09-14 as suite `tabarena-2026-09-11` (`tabldm_method_metadata`,
  r2://tabarena/cache/artifacts/tabarena-2026-09-11/methods/Xiaomi-TabLDM) and registered in the
  arena collection as a verified model.

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
    benchmark_name="xiaomitabldm_11092026",
    model_jobs=[
        ModelJob(
            models=("Xiaomi-TabLDM", 0),
            name="gpu",
            resources={
                "num_gpus": 1,
                # The gpu_partition's VRAM in GB (gpurtxpro6000flex -> RTX PRO 6000 -> 96).
                # AutoGluon budgets parallel bagging folds against this figure instead of the
                # node RAM, so the check reflects the card the tensors actually live on.
                "fake_memory_for_estimates": 96,
            },
        ),
    ],
    task_subset=TaskSubset(),  # the full task set (all splits)
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_mitra_v2_10092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),  # override: log model fits
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),  # override: auto-detect on node
    # bundle_size=10: ten fits per SLURM array task (maintainer's choice on 2026-09-11, replacing the
    # bundle_size=1 array 1142547 cancelled minutes after launch); fewer, longer tasks amortize node
    # provisioning on the flex partition.
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000flex", bundle_size=10),
)
plan.setup_jobs()
```

---

## 2026-09-10 — mitrav2_10092026

- **Model(s):** Mitra-v2 (0 — single default config, no HPO: the frozen fine-tuning recipe is the method)
- **Git SHA:** `52fb5e2f` for 807 of the 816 splits; the 9 `hiva_agnostic` splits ran on `f795c3cc` plus
  the `configure_cuda_allocator` change of the same PR (#520), see notes.
- **Purpose:** First TabArena-v0.1 benchmark of Mitra-v2 (arXiv:2609.04540), Amazon's second-generation
  Mitra tabular foundation model, fine-tuned per bag child under the reference recipe. Supports binary,
  multiclass and regression, so all problem types are included.
- **Notes:** GPU partition `gpurtxpro6000flex` (RTX PRO 6000, 96 GB VRAM, 24 vCPUs per node), 1 GPU,
  `bundle_size=1`, full task set (`TaskSubset()`, all splits: 816 array tasks). `fake_memory_for_estimates=96`
  so the wrapper's static memory estimate is read against VRAM; the wrapper itself pins
  `fold_fitting_strategy="sequential_local"`, so the 8 bag children run one after another on the card.
  The v0.1 default fit budget of 1 h per config was kept: the recipe's 250 s per-child fine-tuning budget
  was designed for it. A first submission on `gpurtxpro6000spotinteractive` was cancelled minutes after
  launch in favour of the flex partition. Main array 16:23 to 02:50 (about 10.5 h wall); throughput was
  bounded by flex node provisioning at about 32 concurrent tasks, with node boots regularly stalling for
  20 to 50 min. APSFailure (50,666 rows, 170 features) is the slowest task: about 3270 s fit (every
  child's fine-tune hits the 250 s budget, two out-of-memory support halvings per child from 16384 to
  4096 rows) plus about 1500 s of test inference, finishing 14 min under the 2 h SLURM wall time.
  `hiva_agnostic` (1413 mostly binary columns, kept in full by the recipe's continuous-fraction gate)
  failed at prediction on all 9 splits, on two attempts: a 13.9 GiB feed-forward activation could not
  be allocated with 59.5 GiB live and 32.8 GiB reserved but fragmented. Fixed without touching the
  recipe by enabling `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` from the wrapper's warm-up
  (`configure_cuda_allocator` in `mitra_v2/model.py`); the 9 splits were resubmitted through the same
  `setup()` (the cache check re-approved exactly those) and completed in 49 to 51 min each (log loss
  0.174 to 0.176, about 2470 s fit). 127 GPU-hours of measured training time in total. Eval on
  `[[], ["binary"], ["multiclass"], ["regression"]]`: Elo 1766 (position 3) on the full leaderboard
  against 1314 for Mitra (v1). Extra dep in the run venv: `autogluon.tabular[mitra]>=1.6,<1.7`;
  `flash-attn` not installed (the reference numbers were calibrated without it). Venv
  `tabarena_mitra_v2_10092026` imports the `add-mitra-v2` checkout of this clone.

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
    benchmark_name="mitrav2_10092026",
    model_jobs=[
        ModelJob(
            models=("Mitra-v2", 0),
            name="gpu",
            resources={
                "num_gpus": 1,
                # The gpu_partition's VRAM in GB (gpurtxpro6000flex -> RTX PRO 6000 -> 96).
                "fake_memory_for_estimates": 96,
            },
        ),
    ],
    task_subset=TaskSubset(),  # full task set (all splits)
    path_setup=PathSetup(
        workspace="/home/lennart_priorlabs_ai/workspace/benchmarking/tabarena_workspace",
        python_path="/home/lennart_priorlabs_ai/.venvs/tabarena_mitra_v2_10092026/bin/python",
    ),
    experiment_bundle=TabArenaV0pt1ExperimentBundle(model_verbosity=2),
    resources_setup=TabArenaV0pt1ResourcesSetup(num_cpus=None, memory_limit=None),
    # bundle_size=1: one fit per SLURM array task (each is ~8 fine-tunes).
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000flex", bundle_size=1),
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
