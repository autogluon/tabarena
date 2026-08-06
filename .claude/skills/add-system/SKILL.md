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
    tags=(),                         # see below
    verified=False,                  # until signed off
)

<system_key>_info = SystemInfo(
    system_cls=<SystemName>SystemModel,
    config_generator=gen_<system_key>,
    method_metadata=<system_key>_method_metadata,
    pip_extra=("<package>==<version>",),
    prefetch_weights=None,           # or the system's weight-warming callable
)
```

`MethodMetadata.system(...)` fixes `method_type="baseline"` (a system's raw results are recorded that way by the runner) and sets `method_class="system"`. `SystemInfo` asserts the latter, so a misdeclared system fails at import instead of misclassifying on the leaderboard.

### Choosing tags

`tags` is what lets a reader rule a system out, and it decides which entrant pools it competes in (`evaluation/entrants.py`). Only two values exist; ask the user when either is unclear rather than guessing.

| tag | when | effect |
|---|---|---|
| `with-llm` | an LLM is involved anywhere, agents included | excluded from the `models` and `systems_open` pools |
| `closed-source-api` | runs behind a remote API we cannot inspect | only in `systems_all` |

No tags means open-source, local and LLM-free, which is the common case (AutoGluon, LightAutoML, FLAML, TabFM+). The two are independent: an open-source agent is `("with-llm",)`, and a hosted non-LLM predictor is `("closed-source-api",)`.

If the system needs a property neither tag covers, do not invent one inline. Add it to `MethodTag` in `models/_method_metadata.py`, give it presentation in `website_format.TAG_SPECS`, and decide which pools admit it in `evaluation/entrants.py`. All three or none, otherwise it will not render.

## Step 4: the pip extra

Add the system's dependency to the `[project.optional-dependencies]` block in `packages/tabarena/pyproject.toml`, matching `SystemInfo.pip_extra`.

## Step 5: register and verify

- Add the metadata to `tabarena_method_metadata_collection` in `contexts/tabarena/methods.py` once results exist (see the `upload-method` skill for processing and hosting them).
- `pytest tests/tabarena/systems/ -q` — the registry test checks the new system is discovered and declares `method_class="system"`.
- `ruff check` **and** `ruff format --check` on every touched file.

There is no per-system fit test. Verify the wrapper with the quickstart in `examples/advanced/run_quickstart_tabarena_external_system.py`, which runs one config on a single small dataset.

## What happens on the leaderboard

The system appears under the 📊 **System** family, typed from `method_class` rather than from its name, with a chip per tag. It shows up in whichever entrant pools admit it, and its presence changes every other entrant's Elo and Improvability in those pools, since both are measured against the field.
