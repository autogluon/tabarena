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

## 2026-06-30 — tabfm_30062026

- **Model(s):** TabFM (0 — default config only)
- **Git SHA:** `25ed60ff`
- **Purpose:** TabFM (JAX/Flax in-context tabular foundation model, GPU) on TabArena-v0.1,
  single-node GCP GPU run. Supports binary / multiclass / regression, so all problem types
  are included (unlike the regression-only Nori plan).
- **Notes:** GPU partition `gpurtxpro6000spotinteractive`, 1 GPU, `bundle_size=1`. Empty
  search space (`can_hpo=False`), so it runs its single default config (0 random configs).
  No `fake_memory_for_estimates`: TabFM reports `can_estimate_memory_usage_static = False`
  (no static estimate to compare against a budget), unlike the VRAM-faking DenseLight. The
  `google/tabfm-v1-0-0` checkpoint is prefetched from Hugging Face by the registry's
  `prefetch_weights` before the parallel fits. Requires the `tabfm[cuda]` dependency in the
  run venv. **Had to raise the time limit for a few datasets where the model was very small
  but still ran long:** `customer_satisfaction_in_airline` and `APSFailure` → 4h;
  `GiveMeSomeCredit` → 8h. The embedded snippet is a frozen snapshot scoped to
  `GiveMeSomeCredit` at 8h (`name="gpu_test_3"`); the full run used `TaskSubset(subset="lite")`.

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
    benchmark_name="tabfm_30062026",
    model_jobs=[
        ModelJob(
            models=("TabFM", 0),
            name="gpu_test_3",
            resources={"num_gpus": 1, "time_limit": 2 * 60 * 60},
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
    scheduler_setup=GCPSlurmSetup(gpu_partition="gpurtxpro6000spotinteractive"),
)

benchmark_plan.setup_jobs()
```

---

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
