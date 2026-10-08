---
name: add-system
description: Add a new ML *system* to the TabArena benchmark. Use this skill whenever the user wants to integrate a whole pipeline rather than a single model — AutoML frameworks (AutoGluon, LightAutoML, FLAML, auto-sklearn), LLM-driven agents, hosted prediction APIs (TabPFN-3-API), or a model run through a heavier self-managing interface (TabFM+). Triggers on "add X as a system", "benchmark the X AutoML framework", "wrap this API for TabArena", "integrate this agent". Creates the per-system folder (`system.py`, `hpo.py`, `info.py`) and picks the right `method_class` / `tags`. For a single model under TabArena's shared tuning protocol, use `add-model` instead.
argument-hint: <SystemName> [<pip-package>] [<doc-url>]
user-invocable: true
---

# Add a System to TabArena

## Model or system?

Ask this first, because it decides everything else.

- **Model** — one method that TabArena tunes, using the shared search-space protocol and compute constraints. It plugs into AutoGluon, has an `ag_key`, and gets default / tuned / tuned+ensembled variants. Use the **`add-model`** skill.
- **System** — a pipeline that manages its own budget, model selection, tuning and ensembling. TabArena hands it the data and the constraints and records what comes back. AutoML frameworks, agents, hosted APIs, and models run through a self-managing interface all land here.

A useful test: if you would have to invent a search space for it, it is a model. If inventing one makes no sense because the thing does its own searching, it is a system.

## Layout

Every system lives in one folder at `packages/tabarena/src/tabarena/systems/<system_key>/`, mirroring `models/`:

```
systems/<system_key>/
  __init__.py   re-exports the three below
  system.py     the ExternalSystemModel subclass
  hpo.py        the SystemConfigGenerator (which configurations to benchmark)
  info.py       the SystemInfo + its MethodMetadata
```

`systems/_registry.py::discover_systems()` walks these `info` modules into `SYSTEM_REGISTRY`, keyed by `method_metadata.method`. Read `systems/autogluon/` (a framework) and `systems/tabfm_plus/` (a model through a heavier interface) before writing a new one.

Systems stay **out** of the AutoGluon model registry on purpose: no `ag_key`, no search space, and they run through the experiment bundle's `system_experiments=True` mode.

## Step 1: `system.py`

Subclass `ExternalSystemModel` (`tabarena/benchmark/exec_models/external.py`) and implement `_fit_system`, `_predict` and `_predict_proba`. Read that class's docstring for the full argument contract; the parts that matter most:

- Everything the fit needs is **passed in**, never read off `self`: the raw frames, `target_name`, `problem_type`, `eval_metric`, `validation_metadata`, the compute budget (`num_cpus` / `num_gpus` / `memory_limit` / `time_limit`) and the per-split `random_state`.
- `X` is yours to edit in place. There is no validation split; carve your own from `X`/`y` if the system wants one.
- Add `__init__` arguments for the system's settings and forward `**kwargs` to `super().__init__`. Those arguments are what the config generator varies.
- The compute and time budgets are **not** init knobs. They come per split from the runner, so every system is held to the same constraints.
- Add `cleanup` to free files and memory. Only delete a directory you created; an explicitly passed `path` belongs to the caller.

Keep the library import inside `_fit_system` (or the method that needs it), never at module top level, so an install without the extra still imports.

A system's environment work is its own. TabArena adds no system-specific warm-up, persist or checkpoint prefetch code: a system that imports its stack, downloads checkpoints or loads from disk inside its fit or predict is measured the way it ships (the `AutoGluonSystemModel` docstring spells this out for AutoGluon). Two generic hooks exist, documented in the `ExternalSystemModel` docstring:

- Warm-up: a system may declare `warmup_modules = ("yourlib",)` and/or `warmup_torch_device = True`; that is import and CUDA-context work only, and the synthetic dummy fit is off for systems. Do not write per-system warm-up logic.
- `uses_ray`: keep the base default (True) unless no code path of the system can start Ray; the SLURM worker skips the Ray runtime for fits that answer False.

`SystemInfo.prefetch_weights` is a hook for the system's own tooling; the benchmark setup does not call it, and `offline_weights="auto"` stays off while a system is selected because its checkpoints are not prefetched.

## Step 1b: Hosted APIs

A system that calls a remote service (`tags=("closed-source-api",)`) sends the training table and the
test features to code nobody can read, while every TabArena dataset is public. The wrapper is the
only part maintainers can inspect, so it is held to this contract:

- The key comes from an environment variable read at call time; it is never an init argument, a
  config value or a default in the code. The endpoint may default to the public URL.
- A request carries the training features and target, the test features, the target name, the
  problem type, the metric, the time limit and the config. Never anything that names the task or
  the split (dataset name, OpenML ids, fold, repeat) and never a target column next to test rows.
- Prefer separate calls: fit in `_fit_system` (training data only, returns a model handle), predict
  in `_predict` / `_predict_proba`. The fit then never sees the test rows and the timings split the
  way they do for every other entrant. An API that only offers one `fit_predict` call is accepted,
  but its whole fit is timed as inference (`time_train_s` near zero); say so in the PR.
- Forward `random_state` when the API takes a seed, and `time_limit` as the fit budget.
- Bound the retries (transient errors, HTTP 429 and 5xx; never a 4xx) and give the request a
  timeout above `time_limit`, so a dead endpoint fails an item instead of hanging it.
- Record what the server reports in `get_metadata()`: the model or API version, server-side fit and
  predict time, the hardware. The runner stores it in `results.pkl` (`method_metadata`), which is
  how a later reader tells which version of a changing service produced a result.
- Import the HTTP client inside the method that calls it, like any optional dependency.

In `info.py`, `compute` is the provider's hardware (processing accepts the CPU client node that the
raw results record, with a warning), `license` is the terms of service plus the licenses of any
models served behind the API, and `commercial_use` is False when one of those models is
non-commercial: ask the submitter what runs behind the endpoint.

Maintainers also need, before the benchmark run: an evaluation key with quota for every config on
every split (816 per config on TabArena-v0.1, plus about 30 calls for the audit), the concurrency the
provider allows, and the hardware behind the API.

Check the client with `python -P -m tabarena.tools.audit_system --system <SystemName> --offline`: it
lists what the client sends with the network blocked (no key needed, nothing sent). With a key, the
same command without `--offline` probes the API for transduction and label lookup (see the module
docstring).

## Step 2: `hpo.py`

```python
gen_<system_key> = SystemConfigGenerator(
    model_cls=<SystemName>SystemModel,
    name="<SystemName>",          # a system has no ag_name/ag_key, so this is required
    manual_configs=[{}, {"preset": "high_quality"}],
)
```

Each config becomes one benchmarked variant. Prefer a small set of meaningful presets over a search space: the point of a system is that it searches for you.

## Step 3: `info.py`

```python
<system_key>_method_metadata = MethodMetadata.system(
    method="<SystemName>",
    name="<SystemName>",
    suite="tabarena-<YYYY-MM-DD>",   # required, must differ from `method`
    compute="cpu" | "gpu",
    date="<YYYY-MM-DD>",
    date_introduced="<YYYY-MM>",
    reference_url="...",
    license="Apache-2.0",            # what governs using the system, incl. every bundled model's weights
    commercial_use=True,             # False when any bundled component is non-commercial (e.g. TabPFN-3)
    tags=(),                         # see below
    verified=False,                  # until signed off
)

<system_key>_info = SystemInfo(
    system_cls=<SystemName>SystemModel,
    config_generator=gen_<system_key>,
    method_metadata=<system_key>_method_metadata,
    pip_extra=("<package>==<version>",),
    prefetch_weights=None,           # optional hook for the system's own tooling; the benchmark setup does not call it
)
```

`MethodMetadata.system(...)` fixes `method_type="baseline"` (a system's raw results are recorded that way by the runner) and sets `method_class="system"`. `SystemInfo` asserts the latter, so a misdeclared system fails at import instead of misclassifying on the leaderboard.

### Choosing tags

`tags` is what lets a reader rule a system out, and it decides which entrant pools it competes in (`evaluation/entrants.py`). Only two values exist; ask the user when either is unclear rather than guessing.

| tag | when | effect |
|---|---|---|
| `with-llm` | an LLM is involved anywhere, agents included | competes only where the `llm` category is selected |
| `closed-source-api` | runs behind a remote API we cannot inspect | competes only where the `api` category is selected |

A system carrying both tags needs both categories selected, so it never appears on the strength of a property the reader excluded.

No tags means open-source, local and LLM-free, which is the common case (AutoGluon, LightAutoML, FLAML, TabFM+) and puts the system in the `open` category. The two tags are independent: an open-source agent is `("with-llm",)`, and a hosted non-LLM predictor is `("closed-source-api",)`. Each combination of categories is published as its own pool, so a new tag doubles the artifact count.

If the system needs a property neither tag covers, do not invent one inline. Add it to `MethodTag` in `models/_method_metadata.py`, give it presentation in `website_format.TAG_SPECS`, and decide which pools admit it in `evaluation/entrants.py`. All three or none, otherwise it will not render.

## Step 4: the pip extra

Add the system's dependency to the `[project.optional-dependencies]` block in `packages/tabarena/pyproject.toml`, matching `SystemInfo.pip_extra`.

## Step 5: register and verify

- Add the metadata to `tabarena_method_metadata_collection` in `contexts/tabarena/methods.py` once results exist (see the `upload-method` skill for processing and hosting them).
- `pytest tests/tabarena/systems/ -q` — the registry test checks the new system is discovered and declares `method_class="system"`.
- `ruff check` **and** `ruff format --check` on every touched file.

For a hosted API, add the offline audit (Step 1b) to the verification and paste its request outline
into the PR.

There is no per-system fit test. Tests specific to the system, if any, go into `systems/<key>/tests/` (with an empty `__init__.py`), outside the default suite and CI, as for models (add-model, Step 3f). Verify the wrapper with the quickstart in `examples/benchmarking/run_quickstart_tabarena_system.py`, which runs two configs of the demo system on the small datasets' first split (`examples/beyondarena/run_quickstart_beyondarena_system.py` is the BeyondArena counterpart and runs the shipped AutoGluon wrapper).

## Step 6: Report and open the PR

Summarize the new files, the edited files, the chosen `tags` and why, and the open TODOs (results,
`verified`, the `methods.py` registration once artifacts exist).

When asked to open the PR, use `.github/pull_request_template.md`: a two-to-four sentence summary,
everything longer inside the collapsed `<details><summary>Details</summary>` block, the commands run
under Tests. Fill in the "Model or system submission" section for a system (delete the model lines)
and keep the closing contribution line. Do not paste the Report into the PR body; the Report is for
the chat, the PR body is for the reviewers. State the TabArena-Lite (or BeyondArena `core2k`) results
with the hardware and the entry-point script if they exist; if they do not, say so, since a
maintainer will ask (TabArena verifies submitted results by re-running them, it does not benchmark
on request). Questions go through the issue forms in `.github/ISSUE_TEMPLATE/`.

## What happens on the leaderboard

The system appears under the 📊 **System** family, typed from `method_class` rather than from its name, with a chip per tag. It shows up in whichever entrant pools admit it, and its presence changes every other entrant's Elo and Improvability in those pools, since both are measured against the field.
