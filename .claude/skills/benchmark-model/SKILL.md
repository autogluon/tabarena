---
name: benchmark-model
description: Run one already-integrated model on the TabArena benchmark, from a local smoke fit to the cluster run and the evaluated leaderboard. Use this skill whenever a maintainer wants to benchmark a model that is already in the registry, e.g. "benchmark TabM", "run Nori on the cluster", "create a setup/eval script for DenseLight", "launch <model> on TabArena and evaluate it". It first asks whether Claude should drive the run end-to-end here (launch, monitor, evaluate, report) or hand the launch and monitoring to the maintainer, then scaffolds a single `tmp_scripts/run_<model>.py` at the repo root with `smoke`, `setup` and `eval` subcommands sharing one benchmark_name + paths. Defaults to the full TabArena-v0.1 task set with all configs (0 for foundation models without a search space), resolves the run venv to the one importing this checkout, installs the model's pip extra into it, smoke-fits the model locally (on the CPU when no GPU exists, GPU models included), and requires `fake_memory_for_estimates` set to the partition's VRAM for every GPU model (asks when not inferable). In end-to-end mode it launches the sbatch command(s), reports the percentage of tasks left at regular intervals while checking for failed tasks, then runs the eval with all figures for the default subsets and reports the leaderboard with the new model highlighted (also box-labeled in every Pareto figure). Also runs registered systems, and for hosted APIs (`closed-source-api`) audits the client and probes the API for cheating (`tabarena.tools.audit_system`) before a single-node run from an on-demand CPU node. Complements `add-model` (integrate a model) and `upload-method` (publish its results).
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
Copy it and fill the `<...>` and `# EDIT` markers; do not hand-write the structure. A registered
system, and in particular a hosted API, follows [Systems and hosted APIs](#systems-and-hosted-apis)
and its template [`references/run_api_system_template.py`](references/run_api_system_template.py). The progress
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

When `MODEL` is not in the model registry but in `tabarena.systems` (`systems/<key>/`), it is a system:
switch to [Systems and hosted APIs](#systems-and-hosted-apis) now.

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

## Step 1b: Check `_fit` for tuning on its own splits

Read the wrapper's `_fit` and the library calls it makes for any step that tunes or selects on a
split of the training data instead of the `X_val` / `y_val` TabArena passes: a hyperparameter
search, an internal cross-validation or hold-out split, or ensemble or candidate weights solved on
rows held out of `X`. Model submissions may not do this. When you find one, stop before Step 2 and
raise it with the maintainer, quoting the lines and saying what they fit on. List the options from
autogluon/tabarena#637: submit it as a system (`add-system`), use a version or checkpoint without
the step, or change the wrapper so the step runs on the passed `X_val` / `y_val`, the way a
fine-tuning API takes an eval set. Launch nothing until the maintainer decides. A PR description
that mentions a hold-out is the quickest tell (#637: "fits its candidate/ensemble weights on a
single 20% hold-out split"), but read the code, because a library can do it without the PR saying
so. Early stopping on `X_val` is fine, and so is an in-context model that ignores `X_val`.
EXAONE-Tabular's regression hold-out was accepted by oversight and is no precedent.

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

When the model has its own tests (`packages/tabarena/src/tabarena/models/<key>/tests/`), run them
too: `$PY -m pytest packages/tabarena/src/tabarena/models/<key>/tests -q`. CI never runs them, so
this is where they catch a wrapper regression before the cluster does.

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
`sky jobs pool apply -y -p <pool> <pool.yaml>` (the YAML carries the worker count), `sky jobs pool status --all <pool>` (wait
for READY) and `sky jobs launch -y -d --pool <pool> ...` (pool mode). Run them as printed and record
the job ids from `sky jobs queue`. The block ends with the eval reminder and, in pool mode, with
`sky jobs pool down -y <pool>`, which must run when the benchmark is finished (a pool bills while idle).
Do not re-apply the pool YAML with a smaller `workers:` while items still run on the pool: on this
cluster's fork the apply bumps the pool version and replaces every worker, and a managed job recovered
onto a new worker resumes its claim but restarts the item from scratch with a fresh budget
(`#RECOVERIES` in `sky jobs queue --all` increments, `sky jobs logs <id>` shows `resuming own claim`;
the `beyondarena_tfms_22092026` run lost 11.5 h and 6 h of two TabPFN-3.5 fits that way). Leave the
idle workers until the last item ends, or scale only a pool with nothing running; a changed
environment always needs a fresh pool for the same reason.
The block's first line names the launch id, the bucket queue and the bundle count.
Keep a pool name at 22 characters or fewer (`SkyPilotSetup` rejects longer ones): SkyPilot cuts a longer
worker cluster name on GCP to a 23-character prefix plus a 2-character hash, so the workers of one pool, and
of two pools that share the prefix, collide on the `skypilot-cluster-name` label. The first Kumo-Tabular run
(`tabarena-kumotabular-28092026`, 32 workers) sat at 19 READY for half an hour that way while every VM billed,
and a second pool next to it collided with the first.

## Step 6: Monitor the run (end-to-end mode)

Watch each array with the progress script through the `Monitor` tool, one monitor per array,
`persistent: true` (runs span hours):

```bash
.claude/skills/benchmark-model/references/slurm_progress.sh <job_id> <K+1> \
  --results-dir <WORKSPACE>/output/<benchmark_name>/data --expected-results <existing + N> \
  --log-dir <WORKSPACE>/slurm_out/<benchmark_name>/<job_id> --interval 900
```

For a SkyPilot launch use `references/sky_progress.sh <queue_uri> <n_bundles> [--launch <launch_id>]
[--pool <pool>] [--interval 900]` instead (both values are on the first line of the printed command block). It counts
the `done/` and `failed/` markers in the bucket queue, lists `sky jobs queue` for the launch, and exits
when every bundle is done or no worker job is left. Failed items are listed with their coordinates;
their logs are at `<run_uri>/logs/<launch_id>/<bundle>_<item>.log` (`gcloud storage cat`). A `setup`
relaunch re-enumerates only the missing items into a fresh queue, exactly like SLURM, but only after
the previous launch has drained (`sky_progress.sh` prints `DONE` or `WORKERS GONE`) or was cancelled:
the cache check sees only synced results, so items still in flight would be enumerated again and
fitted twice. `setup` warns about such a launch (in its log and in the printed command block); do not
launch over it. Editing the checkout or the venv while a launch runs is safe, the workers use the
environment and batch staged in the bucket.

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
| `PENDING` with reason `launch failed requeued held` | the spot node failed to boot; SLURM requeued the task but holds it until someone releases it | `scontrol release <job>_<task>`; a loop over `squeue -r -o "%i %r"` every few minutes when the partition keeps losing nodes |
| `LocalEntryNotFoundError`, `PretrainedWeightsUnavailableError`, `WeightsUnavailableError` | the job ran with `offline_weights` and a checkpoint was not in the shared cache | re-run `setup` (its head-node prefetch fills the cache) or set `offline_weights=False` on the plan for that run |
| `##### item FAILED` lines with `##### bundle summary: ok=N failed=M` | one item of the bundle failed; the array task continues with its siblings and exits non-zero at the end | count the `results.pkl` files, not the task state: a `FAILED` task may have completed most of its items, and only the failed ones are missing |
| a pool stuck below its size with workers in `STARTING` after their setup log ends `SUCCEEDED`, the controller log (`sky jobs pool logs --controller --no-follow --tail 300 <pool>`) repeating `Found 2 node(s) with the same cluster name tag`, `sky_progress.sh --pool <pool>` printing `DUPLICATE VM` | two workers share one GCP cluster name because the pool name is too long (Step 5); the prober fails on every pass and promotes no worker | do not delete one of the VMs, it may belong to the other worker; cancel the launch's worker jobs, take the pool down, re-run `setup` with a pool name of at most 22 characters and launch again (finished items stay cached) |
| an item running far past a finished sibling's time, `#RECOVERIES > 0` for its job in `sky jobs queue --all`, `resuming own claim` in `sky jobs logs <id>` | the worker was replaced (a spot preemption, or a pool re-apply, see Step 5) and the item restarted from scratch; not a stall | nothing, apart from noting the restart in the log entry; compare the fit times of finished siblings before calling an item stuck |
| `TimeLimitExceeded` from `_get_fold_time_limit` after thousands of `CUDACachingAllocator ... memory allocation failed` or `auto batch skip` lines, one to three children fitted in hours | a library that survives GPU out-of-memory by shrinking its batches and crawls until AutoGluon projects the remaining folds past the limit: a memory failure that never raises | treat it as out-of-memory and cap the in-context table in the wrapper. Measure rows and columns separately before choosing the cap: LimiX-2 passed 38k x 318 in 4 h but not 53.6k x 243 or 7.9k x 1652 (the same 13M cells), so a rows x columns budget is the wrong proxy; prefer the library's own knobs (`test_batch_size`, `max_num_rows`), engage the cap only above a shape that fails, and rerun every affected table with the final code |

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

Keep one array per benchmark run. When the plan changes while an array runs (a dataset group moves
to another scheduler, the bundle size or time limit changes, a fix needs a relaunch), cancel the
running array first and run `setup` once more so a single new array holds everything still
missing. Do not submit a second array next to a running one: the two overlap unless the datasets
were split by hand, three arrays need three sets of monitors and requeue loops, and a second
`setup` under the same `ModelJob` name overwrites the `slurm_run_data_<benchmark>_<name>.json`
that the running tasks read when they start, so they would run the wrong items. Cancelling is
cheap: every finished item already has its `results.pkl`, so only the item each task is on is lost,
and the cache check folds the rest into the new array. The same rule holds for SkyPilot launches:
let a launch drain or cancel it before the next `setup`.

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
subset under `subsets/<subset>/`, `tabarena_leaderboard.csv`, `results_per_split.csv` (one row per dataset,
split and method), the `tuning-impact-elo*` bar plots,
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

The report goes to the chat. Never comment on the PR or edit its description on your own initiative,
not even for the stage-2 results comment the PR template describes: draft the text, show it to the
maintainer, and post it only after they say so.

### CPU models: check the fit times before they replace hosted results

A re-run of a CPU model reproduces its Elo but not necessarily its fit times: CatBoost and EBM re-run on
`n4-standard-16` spot nodes in September 2026 matched the July suite's Elo exactly while their median fit
time per 1K rows came out 1.5 to 1.6 times higher, and the results were not hosted for that reason. Before a
CPU rerun takes over a collection slot, compare its `median_time_train_s_per_1K` with the suite it replaces
and explain any gap. The open questions to settle, with tests rather than assumptions: whether the tree
boosters should be timed with the physical cores only (the 16 vCPUs of an `n4-standard-16` are 8 physical
cores with hyper-threads; `num_cpus=None` resolves to all 16) or with every vCPU, whether their thread
count should follow that choice, and which machine and CPU count the hosted suite was timed on (record
both in the log entry of every CPU run). The same run type on GPU nodes is unaffected: the foundation
models re-run under the new pipeline got faster, as the pipeline change predicts.

## Step 8: Record the run and hand over

Append the run to `packages/tabflow_slurm/BENCHMARK_LOG.md` (newest first, template at the top of
that file): model and config count, the git SHA the jobs ran on, the validation protocol key the
context enforced (`8x1` for TabArena), the purpose, the notes (partition or SkyPilot infra,
accelerator, workers and pool name, the env manifest hash and repo shas from `launch.json`,
bundle size, VRAM setting, wall time, failures and their fixes, extra deps, the venv), and the
verbatim `setup()` plan as run. The log is committed even though the script is not. In hand-off mode
offer the entry; in end-to-end mode write it.

Next in the lifecycle is the `upload-method` skill, pointed at `<WORKSPACE>/output/<benchmark_name>/data`.

## Systems and hosted APIs

A system (`packages/tabarena/src/tabarena/systems/<key>/`) runs through the same plan with four
differences. The job entry is its generator, `ModelJob(models=(gen_<key>, 0))`, which runs every
manual config (`<Name>_c1_default`, `_c2_default`, ...) and has no search space. The bundle carries
`system_experiments=True`. The plan sets `prefetch_model_weights=False`. There is no registry smoke
fit: `smoke` runs the system through the official pipeline on the first split of three small tasks.
`EvalMethod("<System>")` resolves through the system registry; with several configs give each its
own `EvalMethod(ag_name_override=f"{name}_c{i}", display_name_override=...)` so each is labelled, and
remember for `upload-method` that every config is hosted as its own `MethodMetadata` (see
`systems/autogluon/info.py`). Local systems (AutoGluon, TabFM+) otherwise follow Steps 2 to 8 with
the model template.

A hosted API (`tags=("closed-source-api",)`) uses
[`references/run_api_system_template.py`](references/run_api_system_template.py) (`audit`, `smoke`,
`setup`, `eval`) and these steps instead of Steps 4 to 6. The API gets the training table and the
test features, and every TabArena dataset is public, so nothing is launched before the client has
been read and the API probed.

1. Read the client. Read `system.py` for what leaves the process and how: endpoint, key handling,
   payload, retries, timeouts, which fit inputs it forwards. Then run `audit --offline`: the
   static scan plus the first request the client tries, with the network blocked, so no key is
   needed and nothing is sent (the template sets a placeholder key for wrappers that read it before
   building the request). Report the request outline to the maintainer: hosts, JSON keys, table
   shapes and columns. A task identifier (dataset name, task id, fold), a target column outside the
   training table, or a host other than the provider's endpoint stops the run.
2. Ask for what only the maintainer knows, in one `AskUserQuestion` when it is not in the context:
   whether the evaluation key is available, and the concurrency the provider allows (`NUM_WORKERS`).
   The key lives in an environment variable in the maintainer's shell. Never write it into a script,
   a file in the repo, the job JSON, a log entry or a PR. Without the key, stop after the offline
   audit and `setup` and say what is pending.
3. With the key, run `audit` (the live probes: transduction, relabeled and shuffled labels, feature
   jitter against a local reference, the time limit; about eight calls on each of four datasets).
   Set `SUBMITTED_RESULTS` to the submitter's self-reported TabArena-Lite per-split CSV for each config
   (`{"<Name>_c1_default": "<path>"}`); `audit` compares each with the hosted methods. Read only CSVs
   from a submitter's release; never unpickle their `.pkl` files. A `FAIL` stops the
   run: quote the line and raise it with the maintainer. A `probe` FAIL means the system raised; the
   line names the call that failed and the calls served before it. An outage fails the first call of
   a dataset. When only the decoy, relabeled or shuffled calls fail, rebuild that request locally
   (column types, value ranges) and look for an input the server cannot handle before reading it
   as a refusal of the probes, then rerun the dataset; the Chakra-Tab audit's 503s on decoys came
   from `uint8` columns turned into negative floats, which the decoys now avoid. Read the details of
   every `WARN` and report them. The JSON lands in `tmp_scripts/eval_output/<benchmark_name>/`; name it in the log entry.
4. `smoke` (key needed): every config on three small tasks through the official pipeline, one call
   each. Report the errors and the train and inference times. With `SUBMITTED_RESULTS` set, `smoke`
   then compares each config's errors with the self-reported ones on the same splits (TabArena-Lite
   is split 0, which the smoke runs): `compare_reproduced_results` in `tabarena.tools.audit_system`,
   one `SMOKE [verdict] reproduced` line per config and `reproduction_smoke.json`. A system that
   seeds its fit from the split matches the submitted errors up to float noise (relative 1e-4); an
   unseeded one passes within noise. A `FAIL` (our errors more than 10% above the submitted ones at
   the median) stops the launch: the submitted numbers came from something other than what the API
   serves us, so raise it with the maintainer. Report every `WARN` (median off by more than 2%, or a
   split outside 0.8x to 1.25x).
5. `setup` writes the job JSON and prints one `sbatch` command (`SlurmSingleNodeSetup`: an on-demand
   `cpuhighmem16` node, `NUM_WORKERS` items in flight; the client needs no GPU, and a spot node would
   restart a multi-day run). `--scheduler array` instead spreads the items over CPU nodes with
   `array_job_limit=NUM_WORKERS`. Default to hand-off for the launch: the maintainer exports the key
   in the shell that runs `sbatch` (the job inherits it). Resubmitting the same command resumes.
6. Monitor the single job with `squeue -j <id>`, the `results.pkl` count under
   `<WORKSPACE>/output/<benchmark_name>/data` against the expected total, and the runner log
   `<WORKSPACE>/slurm_out/<benchmark_name>/<id>/run.out` (one START and one OK / FAILED line per item;
   each item's own log is in `items/`). HTTP 429 or 5xx failures mean too many requests in flight:
   lower `NUM_WORKERS`, then resubmit after the job ends.
7. `eval` as in Step 7. It ends with the same comparison on every split the submission covers (all
   51 TabArena-Lite tasks, `EVAL [verdict] reproduced` lines and `reproduction_eval.json`); report
   the verdict next to the leaderboard. Say in the report that API timings are wall-clock at the client, including
   the network and the provider's queue, and that an API which fits and predicts in one call records
   its whole fit as `time_infer_s` with `time_train_s` near zero, so its Pareto position against
   locally timed methods is not comparable.
8. The log entry (Step 8) adds the audit and reproduction verdicts and JSON paths, `NUM_WORKERS`, the partition, and
   the API version or model id when the results record one (`method_metadata` in `results.pkl`).

`upload-method` accepts the client node's `compute="cpu"` against the provider hardware that a
`closed-source-api` system declares in `info.py` (a warning, not an error).

## Notes

- The scripts target TabArena-v0.1 (`TabArenaContext`, `TabArenaV0pt1ExperimentBundle`,
  `TabArenaEvalConfig`). BeyondArena is a different eval shape (`BeyondArenaContext`,
  `BeyondArenaEvalConfig`, `BenchmarkRun` comparisons); adapt
  `packages/tabflow_slurm/experiments/run_beyondarena.py` for that instead.
- SLURM tasks import the checkout live over the shared filesystem (SkyPilot workers use the archive
  staged at `setup`). Do not pull, rebase or edit package code in that checkout while an array runs;
  a task that starts during the change fails on a half-updated registry. Test-only edits are safe.
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
  Workers never fetch datasets or weights from OpenML or the Hub: `setup` seeds the tasks it materialized
  (upgrading legacy BeyondArena pickles in place) and the models' weights (prefetched once on the head
  node with its own token) into the shared prefix `<bucket>/tabarena/cache` (`dataset_cache_uri`),
  verifies it and prints two summaries; the worker pulls the weights at start and a dataset's entries
  before its first item, and loads with `HF_HUB_OFFLINE=1` when every model is seeded. A `not cached on
  this node`, `MISSING remotely` or `not seeded` line means the workers download those themselves (gated
  weights then need `HF_TOKEN` in `secrets`).
- Spot partitions preempt; requeued tasks show up as `requeued` in the progress line and are not
  failures. Throughput on `gpurtxpro6000flex` is bounded by node provisioning (about 30 concurrent
  tasks was typical), so a full GPU run of a foundation model takes around half a day.
