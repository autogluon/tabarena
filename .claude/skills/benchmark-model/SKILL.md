---
name: benchmark-model
description: Scaffold the setup + eval scripts to run one model on the TabArena benchmark cluster. Use this skill whenever a maintainer wants to benchmark an already-integrated model — e.g. "benchmark TabM", "run Nori on the cluster", "create a setup/eval script for DenseLight", "I want to launch <model> on TabArena and evaluate it". Generates a single `tmp_scripts/run_<model>.py` with `setup` and `eval` subcommands that share one benchmark_name + paths (so they cannot drift), auto-filling GPU/CPU, eval subsets, the install reminder, foundation-model prefetch from the model's registry `info.py`, and — mandatory for GPU models — the VRAM budget (`fake_memory_for_estimates`; asks the user when the partition's VRAM is not inferable). Complements `add-model` (integrate a model) and `upload-method` (publish its results).
argument-hint: <ModelRegistryName> [<benchmark_name>] [<n_configs>]
user-invocable: true
---

# Benchmark a Model on TabArena

This skill scaffolds the throwaway experiment scripts a maintainer runs to launch **one already-integrated model** on the SLURM cluster and then evaluate it. It replaces the old two-file pattern (`run_setup_<model>.py` + `run_eval_<model>.py`) with **one** `tmp_scripts/run_<model>.py` exposing `setup` and `eval` subcommands.

**Why one file:** setup and eval must use the *same* `benchmark_name` and `PathSetup` (`workspace` + `python_path`) — the eval side reads the raw results from `path_setup.get_output_path(benchmark_name)/data` that setup wrote. Splitting them across two files makes that identity drift. Here it is defined once at the top (`BENCHMARK_NAME`, `WORKSPACE`, `PYTHON_PATH`, `MODEL`) and read by both.

The generated file is a **verbatim, self-contained** plan (no hidden helpers) so its `setup()` body can be copied into `packages/tabflow_slurm/BENCHMARK_LOG.md`, and it lands in gitignored `tmp_scripts/`.

The golden template is [`references/run_benchmark_template.py`](references/run_benchmark_template.py) — copy it and fill the `<...>` / `# EDIT` markers. Do not hand-write the structure.

## Step 0: Gather inputs

Parse `$ARGUMENTS`. Collect (ask only for what's missing or a genuine judgment call):

| Input | Example | Default / source |
|---|---|---|
| `MODEL` | `"TabM"` | **required** — the registry name (see Step 1) |
| `BENCHMARK_NAME` | `"tabm_26062026"` | `<model-lowercased>_<DDMMYYYY>` using today's date |
| `NUM_CONFIGS` | `0`, `25`, `"all"` | `0` if the model has no HPO search space, else ask (foundation models → `0`; tunable models → `"all"` or a capped int under compute limits) |
| task scope | full / lite / regression / dataset names | `TaskSubset()` — the full task set (all splits); `TaskSubset(subset="lite")` for a quick first-split smoke run |
| `gpu_partition` | `"gpurtxpro6000spotinteractive"` | that partition (the current default GPU node); ask if a bigger GPU is needed |
| `fake_memory_for_estimates` | `96` | **required for every GPU model** — the partition's VRAM in GB (see Step 1a); **ask the user if you cannot determine the VRAM from context** |
| per-run overrides | `time_limit` | none unless the model/partition needs them |

`WORKSPACE` and `PYTHON_PATH` default to the values in the template (the shared cluster workspace + the `tabarena_18062026` venv). Keep them unless the user names a different venv/workspace.

## Step 1: Introspect the model registry (auto-fill the mechanical bits)

Given `MODEL`, read the model's contribution under `packages/tabarena/src/tabarena/models/<key>/` and derive:

| Derived value | Where to read it | Drives |
|---|---|---|
| **compute** (`"cpu"`/`"gpu"`) | `info.py` → `ModelDescriptor(compute=...)` / `MethodMetadata.compute` | `resources={"num_gpus": 1}` + `name="gpu"` for GPU; drop the override + `name="cpu"` for CPU |
| **problem types** | `model.py` → `supported_problem_types()` (a classmethod returning a subset of `["binary","multiclass","regression"]`, or `None`/**absent** = all types) | the eval `subsets`: all-types → `[[], ["binary"], ["multiclass"], ["regression"]]` (`[]` = the full set / overall leaderboard); regression-only (e.g. Nori) → `[["regression"]]` and scope setup with `task_subset=TaskSubset(subset="regression")` |
| **HPO search space** | `info.py` → `search_space` (a `gen_<key>` generator); empty/absent ⇒ no HPO | default `NUM_CONFIGS` (foundation models with no real search space → `0`) |
| **pip extra** | `info.py` → `ModelInfo(pip_extra=...)` | the "install into the run venv" reminder in the docstring + Step 4 |
| **weights prefetch** | `info.py` → `ModelInfo(prefetch_weights=...)` (not `None` ⇒ foundation model) | a docstring note that the checkpoint is fetched from HF by the registry before the fits (no per-script action) |
| **static memory estimate** | `model.py` → `_estimate_memory_usage_static` / `can_estimate_memory_usage_static` | whether `fake_memory_for_estimates` can actually cap fold-parallelism (Step 1a caveat) |

Prefer **reading these files** over importing the model (no optional deps needed). If the venv already has the model installed, you may confirm quickly with:
`<PYTHON_PATH> -c "from tabarena.models.utils import get_model_info_from_name as g; i=g('<MODEL>'); print(i.method_metadata.compute, i.pip_extra, i.prefetch_weights, i.model_cls.supported_problem_types())"`

## Step 1a: GPU model ⇒ always set `fake_memory_for_estimates`

Every **GPU** job must carry `"fake_memory_for_estimates": <VRAM_GB>` in its `ModelJob.resources`.
AutoGluon budgets parallel bagging folds by comparing the model's memory estimate against the
reported memory limit (node RAM by default) and splits the GPU only fractionally — **VRAM is never
accounted**. On a RAM-rich node, 8 folds co-schedule on one card and OOM it (this killed every
APSFailure fit of the first `tabm_26062026` TabM run: ~12 GiB VRAM × 8 folds on a 95 GiB card).
Reporting the VRAM as the budget makes the same scheduling arithmetic cap folds by VRAM; RAM-wise
it only makes the budget more conservative, which is safe on the VRAM<RAM nodes we use.

- **Determine `<VRAM_GB>` from context**: the chosen `gpu_partition` (`gpurtxpro6000spotinteractive`
  → RTX PRO 6000 → `96`), the user's request, or the node notes in
  `packages/tabflow_slurm/src/tabflow_slurm/setup/resources.py` (`BeyondArenaResourcesSetup`:
  40/80/96 GB nodes). **If the partition's VRAM cannot be determined from context, ask the user —
  do not guess.**
- **CPU models: never set it** (their estimate must be compared against real RAM).
- **Caveat** — the cap works *through the model's estimate*: if
  `can_estimate_memory_usage_static=False` (e.g. TabFM, TabSwift), AutoGluon falls back to a small
  data-size estimate and fold-parallelism stays at 8 regardless. Still set the value, but tell the
  maintainer to sanity-check per-fold VRAM × 8 (or pin `num_folds_parallel` via
  `ag_args_ensemble`) before launching.

## Step 2: Generate `tmp_scripts/run_<model>.py`

Copy `references/run_benchmark_template.py` to `tmp_scripts/run_<model-key>.py` and fill every marker: the module docstring notes, `BENCHMARK_NAME`, `MODEL`, `NUM_CONFIGS`, the `ModelJob` resources (GPU: `num_gpus` plus the **mandatory** `fake_memory_for_estimates` from Step 1a; any `time_limit`), `task_subset`, `gpu_partition`, and the eval `subsets`. The template defaults to the **full** task set (`TaskSubset()`) and emits both PDF and PNG figures (`figure_file_type=("pdf", "png")`); narrow `task_subset` and `subsets` to `lite` for a quick smoke run. Remove the guidance comments that don't apply so the result reads like the existing per-model scripts. Keep it importable and lint-clean (`from __future__ import annotations`, 120-col).

For a **CPU** model: drop `resources={"num_gpus": 1}`, set `name="cpu"`, and use `scheduler_setup=GCPSlurmSetup(cpu_partition=...)`. For a **local, no-SLURM** run, swap `GCPSlurmSetup` for `LocalSequentialSetup(continue_on_error=True)` and `python_path=sys.executable` (see `packages/tabflow_slurm/experiments/run_tabarena_v0pt1_local.py`).

## Step 3: Report the workflow + reminders

Give the maintainer the two commands and the pre-req:

```
# 1. one-time: install the model's extra into the run venv (if pip_extra is non-empty)
<PYTHON_PATH> -m pip install <pip_extra...>
# 2. launch on the cluster
<PYTHON_PATH> tmp_scripts/run_<model>.py setup
# 3. after the run completes, evaluate
<PYTHON_PATH> tmp_scripts/run_<model>.py eval
```

Then offer to record the run in `packages/tabflow_slurm/BENCHMARK_LOG.md` (newest-first entry, current git SHA, the verbatim `setup()` body, and the *why* in the notes) — that log is committed even though the script is not.

## Notes

- The scripts target **TabArena-v0.1** (`TabArenaContext` + `TabArenaV0pt1ExperimentBundle` + `TabArenaEvalConfig`). BeyondArena is a different eval shape (`BeyondArenaContext`, `BeyondArenaEvalConfig`, `BenchmarkRun` comparisons) — for that, adapt `packages/tabflow_slurm/experiments/run_beyondarena.py` instead.
- `benchmark_name` is a cache key: reuse it unchanged across reruns/eval; change it only for a genuinely new run.
- `ModelJob(name=...)` groups jobs by hardware — models with the same `name` share one `sbatch` command. Give GPU and CPU models different names to split them.
