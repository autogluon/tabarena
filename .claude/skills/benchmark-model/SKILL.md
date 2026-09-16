---
name: benchmark-model
description: Run one already-integrated model on the TabArena benchmark, from a local smoke fit to the cluster run and the evaluated leaderboard. Use this skill whenever a maintainer wants to benchmark a model that is already in the registry, e.g. "benchmark TabM", "run Nori on the cluster", "create a setup/eval script for DenseLight", "launch <model> on TabArena and evaluate it". It first asks whether Claude should drive the run end-to-end here (launch, monitor, evaluate, report) or hand the launch and monitoring to the maintainer, then scaffolds a single `tmp_scripts/run_<model>.py` at the repo root with `smoke`, `setup` and `eval` subcommands sharing one benchmark_name + paths. Defaults to the full TabArena-v0.1 task set with all configs (0 for foundation models without a search space), resolves the run venv to the one importing this checkout, installs the model's pip extra into it, smoke-fits the model locally (on the CPU when no GPU exists, GPU models included), and requires `fake_memory_for_estimates` set to the partition's VRAM for every GPU model (asks when not inferable). In end-to-end mode it launches the sbatch command(s), reports the percentage of tasks left at regular intervals while checking for failed tasks, then runs the eval with all figures for the default subsets and reports the leaderboard with the new model highlighted (also box-labeled in every Pareto figure). Complements `add-model` (integrate a model) and `upload-method` (publish its results).
argument-hint: <ModelRegistryName> [<benchmark_name>] [<n_configs>]
user-invocable: true
---

# Benchmark a Model on TabArena

This skill takes **one already-integrated model** from "it is in the registry" to "here is its
TabArena-v0.1 leaderboard". It scaffolds a throwaway script, installs what the model needs into
the run venv, smoke-fits the model on this machine, launches the SLURM run, follows it to the end,
evaluates the results and reports the leaderboard with the new model highlighted. The maintainer
chooses up front whether Claude drives the cluster part or hands it over.

The script is **one** file, `tmp_scripts/run_<model>.py` at the repo root, with three subcommands:

| Subcommand | What it does | Where it runs |
|---|---|---|
| `smoke` | fits the model on AutoGluon's toy datasets (the registry smoke test without its GPU skip) | this machine, CPU when there is no GPU |
| `setup` | materializes tasks, checks the cache, writes the `JobBatch` + job JSON, prints the `sbatch` command(s) | the head node (this machine) |
| `eval` | post-processes the raw results, builds the leaderboards, writes every figure for the default subsets | this machine |

`setup` and `eval` must use the *same* `benchmark_name` and `PathSetup` (`workspace` +
`python_path`) because `eval` reads what `setup`'s jobs wrote. Both are defined once at the top of
the file (`BENCHMARK_NAME`, `WORKSPACE`, `PYTHON_PATH`, `MODEL`, `NUM_CONFIGS`), so they cannot
drift. The file is self-contained (no hidden helpers) so its `setup()` body can be pasted into
`packages/tabflow_slurm/BENCHMARK_LOG.md`, and it lives in `tmp_scripts/`, which `.gitignore` excludes.

The golden template is [`references/run_benchmark_template.py`](references/run_benchmark_template.py).
Copy it and fill the `<...>` and `# EDIT` markers; do not hand-write the structure. The progress
watcher for the cluster run is [`references/slurm_progress.sh`](references/slurm_progress.sh).

## Step 0: Gather inputs and choose the mode

Parse `$ARGUMENTS`. Collect the inputs below; ask only for what is missing or a genuine judgment
call, and state the defaults you took in the plan.

| Input | Example | Default / source |
|---|---|---|
| `MODEL` | `"TabM"` | required, the registry name (Step 1 verifies it) |
| `BENCHMARK_NAME` | `"tabm_26062026"` | `<model key lowercased, no separators>_<DDMMYYYY>` with today's date |
| `NUM_CONFIGS` | `"all"`, `0` | `"all"` (default config plus the full HPO search space) when the model has a search space; `0` when it has none (foundation models with a frozen recipe). Only a capped int when the maintainer asks for one. |
| task scope | full / lite / regression | `TaskSubset()`, the full task set, all splits. `TaskSubset(subset="lite")` only when the maintainer asks for a first-split trial. A regression-only model gets `TaskSubset(subset="regression")`. |
| GPU partition | `"gpurtxpro6000flex"` | the `GCPSlurmSetup` default (RTX PRO 6000, 96 GB). `gpurtxpro6000spotinteractive` is the same card on spot capacity. Ask only when a bigger card is needed. |
| CPU partition | `"cpun416mtspotinteractive"` | 16 vCPUs / 64 GB, the partition the CPU tree boosters were timed on; `bundle_size=2` |
| `fake_memory_for_estimates` | `96` | required for every GPU model: the partition's VRAM in GB (Step 1a). Ask when it cannot be determined from context. |
| `PYTHON_PATH` | `~/.venvs/tabarena_<...>/bin/python` | the venv whose `tabarena` imports **this** checkout (Step 2). Never assume a name. |
| `WORKSPACE` | the shared cluster workspace | the template value unless the maintainer names another |
| `--scheduler` | `slurm`, `skypilot`, `skypilot-pool` | `slurm` (the GCP SLURM cluster). `skypilot` runs the same plan as SkyPilot managed jobs on their own spot VMs, `skypilot-pool` on a SkyPilot job pool (venv built once per worker; better for many short bundles and for BeyondArena). Only when the maintainer asks for SkyPilot; it needs the cluster's `sky` CLI with `SKYPILOT_API_SERVER_ENDPOINT` set (login nodes preset it), `sky check gcp`, and `HF_TOKEN` in the shell for gated weights. The GPU is the same RTX PRO 6000 (VRAM 96) as the SLURM partition unless the maintainer picks another card. |

Then ask the mode question with `AskUserQuestion`, in the same call as the VRAM question when that
one is needed:

1. End-to-end here (recommended): Claude runs `setup`, launches the `sbatch` command(s), monitors
   the array until it finishes (progress at regular intervals, failed tasks triaged, missing items
   relaunched after a fix), runs `eval`, and reports the leaderboard.
2. Hand-off: Claude does everything up to and including `setup`, then hands over the `sbatch`
   command(s) and the monitoring commands; the maintainer launches and watches the jobs and comes
   back for `eval`.

Do not ask about the config count or the task scope; the defaults above are the benchmark
protocol. Mention them in the plan so the maintainer can object.

## Step 1: Introspect the model registry

Given `MODEL`, read the model's folder `packages/tabarena/src/tabarena/models/<key>/` and derive:

| Derived value | Where to read it | Drives |
|---|---|---|
| compute (`"cpu"` / `"gpu"`) | `info.py`, `MethodMetadata(compute=...)` | GPU: `resources={"num_gpus": 1, "fake_memory_for_estimates": <VRAM>}`, `name="gpu"`. CPU: no resources dict, `name="cpu"`, `GCPSlurmSetup(cpu_partition=..., bundle_size=2)`. |
| problem types | `model.py`, the `_supported_problem_types` class attribute (absent means all three) | the eval `subsets`: all types gives `[[], ["binary"], ["multiclass"], ["regression"]]` (`[]` is the full set); regression-only gives `[["regression"]]` plus `task_subset=TaskSubset(subset="regression")` |
| HPO search space | `info.py`, `search_space` (a `gen_<key>` generator); empty or absent means no HPO | `NUM_CONFIGS`: `"all"` with a search space, `0` without |
| pip extra | `info.py`, `ModelInfo(pip_extra=...)`, and the matching extra in `packages/tabarena/pyproject.toml` | Step 2 installs it |
| weights prefetch | `info.py`, `ModelInfo(prefetch_weights=...)`; not `None` means foundation model (so does a `shared_weights` declaration on the class) | a docstring note; `setup` prefetches the checkpoint on the head node before emitting jobs |
| static memory estimate | `model.py` implements `_estimate_memory_usage_static` | whether `fake_memory_for_estimates` can cap fold parallelism (Step 1a caveat) |
| device selection | `model.py` reads `num_gpus` (falls back to CPU by itself) or a `device` hyperparameter | `SMOKE_EXTRA_HYPERPARAMETERS = {"device": "cpu"}` in the script when the wrapper must be told and this machine has no CUDA device |
| smoke config | `tests/tabarena/models/smoke_configs.py`, `SMOKE_OVERRIDES[<MODEL>]` | the `smoke` subcommand reuses it; nothing to copy |

Prefer reading the files over importing the model. Once the venv from Step 2 is ready you can
confirm in one line:

```bash
$PY -c "from tabarena.models.utils import get_model_info_from_name as g; i=g('<MODEL>'); print(i.method_metadata.compute, i.pip_extra, i.prefetch_weights is not None or getattr(i.model_cls, 'shared_weights', None) is not None, i.model_cls.supported_problem_types())"
```

## Step 1a: GPU model means `fake_memory_for_estimates` is set

Every GPU job carries `"fake_memory_for_estimates": <VRAM_GB>` in its `ModelJob.resources`.
AutoGluon budgets parallel bagging folds by comparing the model's memory estimate against the
reported memory limit (node RAM by default) and never accounts VRAM. On a RAM-rich node eight
folds co-schedule on one card and OOM it; this killed every APSFailure fit of the first TabM run.
Reporting the VRAM as the budget makes the same arithmetic cap folds by VRAM, and RAM-wise it
only makes the budget more conservative, which is safe on the VRAM-smaller-than-RAM nodes we use.

Determine `<VRAM_GB>` from the partition (`gpurtxpro6000flex` and `gpurtxpro6000spotinteractive`
are RTX PRO 6000 cards with 96 GB), the maintainer's request, or the node notes in
`packages/tabflow_slurm/src/tabflow_slurm/setup/resources.py` (`BeyondArenaResourcesSetup`:
40/80/96 GB nodes). If the partition's VRAM cannot be determined from context, ask; never guess.

CPU models never set it (their estimate must be compared against real RAM). The cap works through
the model's estimate: a wrapper without `_estimate_memory_usage_static` (TabFM, TabSwift) falls back
to a small data-size estimate and keeps eight parallel folds regardless. Still set the value, and
tell the maintainer to sanity-check per-fold VRAM times eight, or pin `num_folds_parallel` via
`ag_args_ensemble`, before launching.

## Step 2: Resolve the run venv and install the model's extra

The jobs import the code checked out **here**, so `PYTHON_PATH` must be a venv whose `tabarena`
resolves into this repo. Several venvs under `~/.venvs/` import sibling checkouts; find the right one:

```bash
for v in ~/.venvs/*/bin/python; do echo -n "$v -> "; $v -c "import tabarena, os; print(os.path.realpath(tabarena.__file__))" 2>&1 | tail -1; done
```

Pick the one printing a path under this repo (if `tmp_scripts/README.md` exists it records the
clone's venv). If none does, ask the maintainer whether to create one
(`uv venv --seed --python 3.12 ~/.venvs/tabarena_<clone>_<DDMMYYYY>` then the two installs below).
Set `PY=<that python>` for every command in this skill.

Then install, from the repo root:

```bash
uv pip install --python "$PY" --prerelease=allow -e "./packages/tabarena[benchmark,<extra>]"   # <extra> = the model's pyproject extra
uv pip install --python "$PY" -e ./packages/tabflow_slurm                                     # only if `$PY -c "import tabflow_slurm"` fails
$PY -c "import <the model's library>"                                                          # confirm the extra landed
```

`<extra>` is the `pyproject.toml` extra whose requirement matches `info.py`'s `pip_extra` (usually
the model key, e.g. `tabm`, `mitra_v2`, `nori`; check the file, some differ). If no extra exists,
install the `pip_extra` requirement strings directly with `uv pip install --python "$PY" <req...>`.
Skip the install when the library already imports at the pinned version. Report what was installed.

## Step 3: Generate `tmp_scripts/run_<model>.py`

Create `tmp_scripts/` at the repo root if it does not exist (`.gitignore` lists `/tmp_scripts/`).
Copy `references/run_benchmark_template.py` to `tmp_scripts/run_<model key>.py` and fill every
marker: the module docstring notes, `BENCHMARK_NAME`, `PYTHON_PATH`, `MODEL`, `NUM_CONFIGS`,
`SMOKE_EXTRA_HYPERPARAMETERS`, the `ModelJob` resources (GPU: `num_gpus` plus the mandatory
`fake_memory_for_estimates`; any `time_limit`), `task_subset`, the scheduler (partition,
`bundle_size`), and the eval `subsets`. Remove the guidance comments that do not apply so the result
reads like the existing per-model scripts. Keep it importable and lint-clean
(`from __future__ import annotations`, 120 columns, `ruff check --fix` + `ruff format`).

For a CPU model drop the resources dict, set `name="cpu"` and use
`GCPSlurmSetup(cpu_partition="cpun416mtspotinteractive", bundle_size=2)`. For a local no-SLURM run
swap `GCPSlurmSetup` for `LocalSequentialSetup(continue_on_error=True)` and `python_path=sys.executable`
(see `packages/tabflow_slurm/experiments/run_tabarena_v0pt1_local.py`).

The `eval` half keeps `figure_file_type=("pdf", "png")` so every figure of every default subset is
written in both formats, and `pareto_focus_new_methods=True` so the new model is box-labeled in all
four Pareto figures even when it is not on the front (the front alone is emphasized otherwise).
After `run_eval` the script prints where the run's methods landed in each subset, in Elo order.

`EvalMethod(MODEL)` labels the run's method with the registry's `display_name` in the leaderboard and
every figure (the label the hosted leaderboard uses, e.g. `Xiaomi-TabLDM`) instead of the raw `TA-...`
config type; `display_name_override` replaces it. A re-run of a model that is already hosted needs a
`result_suffix` (e.g. `" [Rerun]"`), which is appended to the label; without it both carry the same
label and `run_eval` prints a warning.

## Step 4: Smoke-fit the model locally

Run the smoke fit before spending cluster time. It is the body of
`tests/tabarena/models/test_all_models.py` for this one model, minus the GPU skip: a GPU model is
fit on the CPU when this machine has no CUDA device (most login and head nodes do not).

```bash
mkdir -p tmp_scripts/logs
$PY tmp_scripts/run_<model>.py smoke > tmp_scripts/logs/<benchmark_name>_smoke.log 2>&1
```

Run it in the background and wait for exit (tree models take a minute or two; foundation models
fine-tuning on the CPU take several). Success ends with `SMOKE OK`. On failure read the traceback:
an import error means Step 2 missed a dependency; a device error means the wrapper needs
`SMOKE_EXTRA_HYPERPARAMETERS = {"device": "cpu"}` (or genuinely cannot run without a GPU, then say
so and ask whether to skip the smoke fit); anything else is a wrapper bug to fix before launching.
Report the outcome and the wall time. The toy fits write an `AutogluonModels/` folder in the
working directory (gitignored); remove the run's subfolders when done.

## Step 5: Set up and launch

`setup` prefetches foundation weights, materializes the tasks, runs the Ray cache check and writes
the job files. It takes minutes; run it in the background and follow the log:

```bash
$PY tmp_scripts/run_<model>.py setup > tmp_scripts/logs/<benchmark_name>_setup.log 2>&1
```

From the log take (a) the `Approved N (experiment, dataset, fold, repeat) items` line, which is the
number of `results.pkl` files the run will add, and (b) the `sbatch --array=0-K%N ... <job.json>`
command(s) after `Run the following ... command(s) to launch the jobs`. `K+1` is the array length.
Also count the results already cached, `find <WORKSPACE>/output/<benchmark_name>/data -name results.pkl | wc -l`,
so the expected total is `existing + N`. Several commands appear when jobs differ in bundle size
or exceed `max_array_size`; each is its own array.

Hand-off mode stops here. Give the maintainer the `sbatch` command(s), the expected totals, the
monitoring one-liner (`references/slurm_progress.sh <job_id> <K+1> --once ...` from Step 6), the
log location `<WORKSPACE>/slurm_out/<benchmark_name>/<job_id>/`, and the `eval` command; then offer
the `BENCHMARK_LOG.md` entry (Step 8).

End-to-end mode: run each `sbatch` command exactly as printed and record the id from
`Submitted batch job <id>`. Tell the maintainer the ids, the array sizes and the partition.

SkyPilot (`--scheduler skypilot` / `skypilot-pool`): `setup` also freezes the run venv, archives the
checkouts it installs and copies the batch plus one task file per bundle into the bucket, then prints
a command block instead of `sbatch`: an endpoint guard, `sky check gcp` (once per machine), then either one
`sky jobs launch -y -d -n <launch_id> --num-jobs N <job.yaml>` (per-job mode) or
`sky jobs pool apply -y -p <pool> --workers N <pool.yaml>`, `sky jobs pool status --all <pool>` (wait
for READY) and `sky jobs launch -y -d --pool <pool> ...` (pool mode). Run them as printed and record
the job ids from `sky jobs queue`. The block ends with the eval reminder and, in pool mode, with
`sky jobs pool down -y <pool>`, which must run when the benchmark is finished (a pool bills while idle).
The block's first line names the launch id, the bucket queue and the bundle count.

## Step 6: Monitor the run (end-to-end mode)

Watch each array with the progress script through the `Monitor` tool, one monitor per array,
`persistent: true` (runs span hours):

```bash
.claude/skills/benchmark-model/references/slurm_progress.sh <job_id> <K+1> \
  --results-dir <WORKSPACE>/output/<benchmark_name>/data --expected-results <existing + N> \
  --log-dir <WORKSPACE>/slurm_out/<benchmark_name>/<job_id> --interval 900
```

For a SkyPilot launch use `references/sky_progress.sh <queue_uri> <n_bundles> [--launch <launch_id>]
[--interval 900]` instead (both values are on the first line of the printed command block). It counts
the `done/` and `failed/` markers in the bucket queue, lists `sky jobs queue` for the launch, and exits
when every bundle is done or no worker job is left. Failed items are listed with their coordinates;
their logs are at `<run_uri>/logs/<launch_id>/<bundle>_<item>.log` (`gcloud storage cat`). A `setup`
relaunch re-enumerates only the missing items into a fresh queue, exactly like SLURM.

Every interval it prints one line: the percentage of array tasks left, the done / failed / running
/ queued / requeued counts, and the `results.pkl` count against the expected total. Each newly
failed task is listed once with its state, exit code and log file. It exits with `DONE ...` when
`squeue` lists nothing for the job (exit code 2 when at least one task failed). Use `--once` for an
on-demand snapshot when the maintainer asks.

Relay each progress line to the maintainer briefly (one sentence with the percentage left and the
counts). Fifteen minutes is the default interval; use 5 minutes for `lite` trials and 30 for
multi-day CPU sweeps. A run that shows no progress across three intervals while tasks are running
deserves a look at a running task's log before reporting it.

When a task fails, read the tail of its `.out` file (`grep -nE "Traceback|Error|out of memory|Killed|TIME LIMIT" <log>`)
and classify:

| Symptom | Meaning | Action |
|---|---|---|
| `NODE_FAIL`, `PREEMPTED`, `REQUEUED` | spot node lost; the submit script sets `--requeue` | nothing, the task re-runs by itself |
| `TIMEOUT` | the fit exceeded `time_limit_per_config x configs_per_job + overhead` | check whether the dataset is a known long tail; a `time_limit` bump or a smaller `bundle_size` for that dataset, then relaunch |
| CUDA out of memory | folds co-scheduled on the card, or one huge table | confirm `fake_memory_for_estimates`; for a single wide table consider `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in the wrapper's warm-up, a smaller support size, or `fold_fitting_strategy="sequential_local"` |
| Python traceback in the wrapper | a bug the smoke fit did not reach | fix it in the wrapper, then relaunch |
| `jq: command not found`, import errors | environment, not the model | fix the venv, then relaunch |
| `LocalEntryNotFoundError`, `PretrainedWeightsUnavailableError`, `WeightsUnavailableError` | the job ran with `offline_weights` and a checkpoint was not in the shared cache | re-run `setup` (its head-node prefetch fills the cache) or set `offline_weights=False` on the plan for that run |
| `##### item FAILED` lines with `##### bundle summary: ok=N failed=M` | one item of the bundle failed; the array task continues with its siblings and exits non-zero at the end | count the `results.pkl` files, not the task state: a `FAILED` task may have completed most of its items, and only the failed ones are missing |

A `FAILED` array task is a bundle with at least one failed item. The log carries one `##### item
FAILED (exit N)` line per failed item and a final `##### bundle summary` line; three consecutive
failures stop the bundle early. Read the traceback of the first failed item, not the last lines of
the file. Predictor artifacts of a job live under `$TABARENA_MODEL_ARTIFACTS_BASE_PATH`
(node-local scratch, removed with the job), so nothing of a failed fit survives on the node; the Ray
worker logs of a failed item are copied to `slurm_out/<benchmark>/<ARRAY_JOB_ID>/ray_logs/task_<i>/`.
Every `results.pkl` records `experiment_metadata["warmup_report"]` and `["timing_audit"]`;
`python -P -m tabarena.tools.audit_warmup --results <WORKSPACE>/output/<benchmark_name>/data`
summarizes them after the run (warm-up failures, packages imported inside the timed sections).

Relaunching is cheap and cache-aware: re-run `setup` (Step 5) after the fix; the cache check
re-approves only the items still missing, and you launch the new, smaller `sbatch` command and
monitor it the same way. Stop the loop and ask the maintainer when a failure needs a design decision
(for example a dataset that cannot fit the card) or when more than a handful of items fail for
the same reason; the Mitra-v2 `hiva_agnostic` case is the reference for such a decision.

The run is complete when every array is `DONE` and the `results.pkl` count equals the expected
total. State both numbers.

## Step 7: Evaluate and report

Run the eval in the background and wait for it (post-processing plus four subsets in two figure
formats takes a while):

```bash
$PY tmp_scripts/run_<model>.py eval > tmp_scripts/logs/<benchmark_name>_eval.log 2>&1
```

Pass the same `--scheduler` as the launch: for SkyPilot the eval first mirrors the bucket's
`output/data` into `<WORKSPACE>/output/<benchmark_name>/data` (nothing local is deleted, so SLURM and
SkyPilot results of one `benchmark_name` merge), then proceeds as usual.

Outputs land in `tmp_scripts/eval_output/<benchmark_name>/`: `leaderboards/<subset>.csv` and, per
subset under `subsets/<subset>/`, `tabarena_leaderboard.csv`, the `tuning-impact-elo*` bar plots,
`winrate_matrix.*` plus `winrate_explorer.html`, the four `pareto_front_*` figures (Elo and
improvability against train and inference time; the new model carries a boxed label in each) and the
two `pareto_front_explorer*.html` pages. The log ends with `Position of this run's methods`.

Report to the maintainer, in this order:

1. Where the new model landed: one line per subset from the position block (position out of N,
   Elo with its interval, imputed share if any).
2. The full-subset leaderboard as a markdown table (the `format_leaderboard` columns the log
   prints: method, Elo, CI, normalized score, rank, improvability, train and inference time per 1K),
   cut to the top ten plus every row of the new model, with the new model's rows in bold. Add a line
   naming the neighbors it displaced or sits between.
3. The figure paths, PNG first, with the Pareto figures called out.
4. Anything the numbers hide: imputed tasks, failed splits that were left out, time-limit hits,
   the `flash-attn` kind of "installed without X" caveat.

## Step 8: Record the run and hand over

Append the run to `packages/tabflow_slurm/BENCHMARK_LOG.md` (newest first, template at the top of
that file): model and config count, the git SHA the jobs ran on, the validation protocol key the
context enforced (`8x1` for TabArena), the purpose, the notes (partition or SkyPilot infra,
accelerator, workers and pool name, the env manifest hash and repo shas from `launch.json`,
bundle size, VRAM setting, wall time, failures and their fixes, extra deps, the venv), and the
verbatim `setup()` plan as run. The log is committed even though the script is not. In hand-off mode
offer the entry; in end-to-end mode write it.

Next in the lifecycle is the `upload-method` skill, pointed at `<WORKSPACE>/output/<benchmark_name>/data`.

## Notes

- The scripts target TabArena-v0.1 (`TabArenaContext`, `TabArenaV0pt1ExperimentBundle`,
  `TabArenaEvalConfig`). BeyondArena is a different eval shape (`BeyondArenaContext`,
  `BeyondArenaEvalConfig`, `BenchmarkRun` comparisons); adapt
  `packages/tabflow_slurm/experiments/run_beyondarena.py` for that instead.
- `benchmark_name` is a cache key: reuse it unchanged across relaunches and `eval`; change it only
  for a genuinely new run. The setup refuses a `benchmark_name` whose cached results were fit under
  another validation protocol than the context enforces; that is a new run, not a relaunch.
- The context asserts the arena's validation protocol (TabArena: 8 folds x 1 set) on every bagged
  experiment; a plan never needs to set fold counts, and a custom protocol is a deliberate
  `official_validation_protocol=False` run that must be named in the log entry and the PR.
- `ModelJob(name=...)` groups jobs by hardware; models sharing a `name` and identical settings share
  one `sbatch` command. Give GPU and CPU models different names to split them.
- `"all"` configs on a CPU model is about 200 configs across 816 splits; `setup` splits the array at
  `max_array_size` (29,999) automatically and `--array=...%100` caps concurrency per array.
- The `smoke` subcommand imports `tests/tabarena/models/smoke_configs.py` from the repo root, so the
  script must stay under `tmp_scripts/` (it derives the root from its own location). Run every
  python entry point (`smoke`, `setup`, `eval`) from the repo root with `python -P`: a directory
  named `autogluon/` on `sys.path[0]` shadows the installed AutoGluon and empties the model registry.
- Before the cluster run, `python -P -m tabarena.tools.audit_warmup --model <Method>` shows whether the
  model's timed fit and predict still import packages or load weights cold; fix that in the wrapper
  (`warmup_modules`, a `shared_weights` declaration) rather than paying it on every item.
- Weights on the cluster: `setup` prefetches the selected models' checkpoints on the head node and,
  when every selected model prefetched (`offline_weights="auto"`), the jobs run with `HF_HUB_OFFLINE=1`
  and `AG_FETCH_PRETRAINED_WEIGHTS=false`, so a cache miss fails the item instead of downloading inside
  the timed fit. The HF cache must be shared between head and compute nodes. Copying the weights onto
  node-local scratch is opt-in (`GCPSlurmSetup(node_staging=NodeStagingSetup(stage_weights=True))`);
  the warm-up pre-loads them untimed either way.
- Two warm-up steps are opt-in until measured on the cluster: the Ray import-only worker pool for CPU
  bags (`TABARENA_RAY_WORKER_WARMUP=1`) and the CUDA kernel probe (`TABARENA_WARMUP_KERNELS=1`). Before
  a campaign, launch one `TaskSubset(subset="lite")` bundle of a CPU booster and one of a GPU model with
  the variable exported in the submitting shell (`--export=ALL` carries it), then read
  `warmup_report.ray` and the fit timings with `audit_warmup --results`; make them defaults only when
  the workers were reused or the probe saved time.
- SkyPilot runs go through the cluster's `sky` CLI and the shared API server (endpoint
  `http://skypilot-api:46580`, preset on login nodes), which resolves `RTXPRO6000:1` to a spot
  `g4-standard-48` and picks the regions itself, so the YAML pins no `infra`. The default card is the
  same RTX PRO 6000 as the SLURM partition (`fake_memory_for_estimates` 96 on both; the template's
  `_scheduler_setup` returns the pair). A preempted worker job is recovered by SkyPilot and resumes
  the bundles it owns; a cancelled launch leaves orphan claims that the next `setup` re-enumerates.
- Spot partitions preempt; requeued tasks show up as `requeued` in the progress line and are not
  failures. Throughput on `gpurtxpro6000flex` is bounded by node provisioning (about 30 concurrent
  tasks was typical), so a full GPU run of a foundation model takes around half a day.
