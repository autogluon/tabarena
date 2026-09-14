---
name: add-model
description: Add a new ML model to the TabArena benchmark system. Use this skill whenever the user wants to integrate a new tabular ML model into TabArena — even if they just say "add X model", "integrate X", "support X", or "wrap X for the benchmark". Creates all required files: the AutoGluon model wrapper, the search-space generator, the per-model `info.py`, and the `pyproject.toml` extra (the model is fit-tested automatically by the registry-driven `test_all_models.py` — no per-model test file). Reads existing similar models for inspiration and optionally fetches documentation URLs to understand the new model's API.
argument-hint: <ModelName> [<pip-package>] [<doc-url>]
user-invocable: true
---

# Add Model to TabArena

This skill integrates a new tabular ML model into the TabArena benchmark.

Every model lives in **one folder** at `packages/tabarena/src/tabarena/models/<ModelKey>/`. That folder contains the wrapper, the HPO generator, and the metadata — and is auto-discovered by `tabarena.models._registry.discover_models()`. There is no separate `benchmark/models/ag/` layout anymore.

Per model, you create up to 5 source files, then edit three existing files. There is no per-model test file — the model is fit-tested automatically by the registry-driven `tests/tabarena/models/test_all_models.py`.

## First: single model or external system?

TabArena has two integration paths; this skill's steps implement the **single model** path, which is
the default:

- **Single model** (default — everything below): one AutoGluon wrapper in `models/<ModelKey>/`,
  fit by TabArena's shared harness. It gets the shared preprocessing + validation protocol, the
  bagged / holdout / outer execution modes, an HPO search space, registry auto-discovery, and
  leaderboard integration.
- **External system**: a self-contained ML system that does its own preprocessing, validation,
  HPO, and/or ensembling — AutoML frameworks, multi-model stacks, LLM/agent pipelines. Systems get
  their own folder and registry, `packages/tabarena/src/tabarena/systems/<key>/` (`system.py` =
  the `ExternalSystemModel` subclass, `hpo.py` = the `SystemConfigGenerator`, `info.py` = the
  `SystemInfo`), and run through a bundle in `system_experiments=True` mode. That path is the
  **`add-system`** skill, not this one. Runnable references:
  `examples/benchmarking/run_quickstart_tabarena_system.py`,
  `examples/beyondarena/run_quickstart_beyondarena_system.py` and
  `examples/advanced/run_async_tabarena_api_system.py` (async/API-driven).

**Ask instead of assuming**: if what the user wants to add looks like a system — it ensembles or
stacks multiple models, runs its own HPO or validation splits, or is described as a "framework",
"AutoML tool", "pipeline", or "agent" — ask which integration they want before proceeding. When
nothing suggests a system, take the single-model path and continue with Step 0.

## Step 0: Gather inputs

Parse `$ARGUMENTS` for the model name. Then collect (ask only for what's missing or unclear):

| Input | Example | Notes |
|---|---|---|
| `ModelName` | `"TabPFN-2.6"` | Human-readable display name |
| `ModelKey` | `"tabpfnv26"` | Snake_case folder/file key (derive from ModelName) |
| `ClassName` | `"TabPFNv26"` | CamelCase class prefix (derive from ModelName) |
| `ag_key` | `"TA-TABPFN-2.6"` | AutoGluon registry key; prefix with `"TA-"` |
| `ag_name` | `"TA-TabPFN-2.6"` | AutoGluon display name; same as `ag_key` with proper casing |
| `pip_package` | `"tabpfn>=7.0.0"` | Pip install spec for pyproject.toml |
| `doc_url` | `"https://..."` | Documentation / GitHub / paper URL |
| `model_type` | `foundation` | `foundation`, `torch`, or `sklearn` |
| `supports_gpu` | `true` | Whether the model uses GPU |
| `problem_types` | `binary,multiclass,regression` | Supported task types |

**Deriving keys**: `"TabPFN-2.6"` → key `"tabpfnv26"`, class prefix `"TabPFNv26"`. `"TabSTAR"` → key `"tabstar"`, class prefix `"TabStar"`. Strip hyphens, lowercase for key; CamelCase for class.

## Step 1: Understand the model API

If `doc_url` was provided, fetch it with WebFetch to understand:
- Import path (e.g., `from tabstar.tabstar_model import TabSTARClassifier`)
- Constructor parameters and their defaults
- `.fit(X, y, ...)` signature
- `.predict()` / `.predict_proba()` signature
- Key hyperparameters to expose

## Step 2: Pick the right base class and reference model

Choose the most similar existing model to read for detailed inspiration:

| Model type | Base class | Read this reference model |
|---|---|---|
| Torch-based foundation / pre-trained GPU (e.g. TabPFN, TabICL) | `AbstractTorchModel` with a `shared_weights` declaration | `packages/tabarena/src/tabarena/models/tabicl/model.py` (the declaration of Step 3g); `tabdpt/` for a library that loads inside its constructor |
| Torch NN trained from scratch (e.g. TabM, RealMLP) | `AbstractTorchModel` | `packages/tabarena/src/tabarena/models/tabm/model.py` |
| Non-torch GPU model (e.g. JAX/Flax like TabFM, or any lib that manages its own device) | `AbstractModel` | `packages/tabarena/src/tabarena/models/tabstar/model.py` |
| CPU / sklearn-like (e.g. KNN) | `AbstractModel` | `packages/tabarena/src/tabarena/models/knn/model.py` |

**`AbstractTorchModel` is only for torch-based models.** Its whole purpose is the torch device
machinery — `get_device()` / `_set_device()` are abstract and the load path calls `torch.cuda.is_available()`
(so a non-torch device string like `"gpu"` would crash it). If the model is **not** torch (JAX/Flax,
or any library that manages device placement itself at the process level, e.g. via
`CUDA_VISIBLE_DEVICES` / `jax.devices()`), inherit **`AbstractModel`** even though it runs on GPU, and
just add the GPU resource attributes (`default_num_gpus`, `minimum_num_gpus`,
`_default_ag_args_ensemble_extra` with `sequential_local`, plus `_more_tags`) — do **not**
implement `get_device`/`_set_device`. `tabstar/model.py` (a GPU foundation model on `AbstractModel`)
is the reference; `tabfm/model.py` is the JAX example.

Read the reference model file now (use the Read tool). Use it as a structural guide — you will adapt rather than copy.

Also read the annotated patterns in `references/model_patterns.md` — it contains templates for `model.py`, `hpo.py`, and `info.py`.

## Step 3: Create new files

Create these files (paths relative to the repo root):

### 3a. `packages/tabarena/src/tabarena/models/{ModelKey}/__init__.py`
Re-export the public symbols so `from tabarena.models.{ModelKey} import ...` works:
```python
from __future__ import annotations

from tabarena.models.{ModelKey}.hpo import gen_{ModelKey}
from tabarena.models.{ModelKey}.info import {ModelKey}_info, {ModelKey}_method_metadata

__all__ = ["gen_{ModelKey}", "{ModelKey}_info", "{ModelKey}_method_metadata"]
```

### 3b. `packages/tabarena/src/tabarena/models/{ModelKey}/model.py`

The AutoGluon wrapper class. Use the template in `references/model_patterns.md` section "Model wrapper template". Key points:
- Start with `from __future__ import annotations`
- Inherit from `AbstractTorchModel` (torch-based models) or `AbstractModel` (CPU models **and non-torch GPU models** — see Step 2: JAX/Flax etc. use `AbstractModel`)
- Set `ag_key`, `ag_name`, `ag_priority = 65`, `seed_name = "random_state"`, and
  `_supported_problem_types = [...]`
- Implement `_fit()` and `_set_default_params()`
- **Declare config as class attributes, not override methods** (AutoGluon 1.6). Read
  `references/model_patterns.md` → "Declare config as class attributes". Overriding
  `supported_problem_types()` is the one AutoGluon actively rejects: `verify_model` raises, so the
  model's smoke test fails. The others (`_get_default_resources`, `get_minimum_resources`,
  `_get_default_ag_args_ensemble`, `_get_default_auxiliary_params`) still work but are the old
  style. Never mutate `self.params` / `self.params_aux` after construction — it raises in 1.7.
- **Honor the `_fit` contract** (read `references/model_patterns.md` → "The `_fit` contract"). The most common review findings on new wrappers are: ignoring the provided `X_val`/`y_val` (and instead auto-splitting a second holdout), ignoring `time_limit`, hardcoding the thread count instead of wiring `num_cpus`, and label-encoding + `fillna(0)` categoricals when the library handles them natively. `models/realmlp/model.py` is the reference for all of these. (In-context-learning foundation models have no train loop / no eval set, so they legitimately ignore `time_limit` + `X_val` — see `sap_rpt_oss`/`tabstar`/`tabfm`.)
- For GPU models: also set `default_resources_physical_cores_only = True`, `default_num_gpus = 1`, `minimum_num_gpus = 1`, and `_default_ag_args_ensemble_extra` (with `fold_fitting_strategy: sequential_local` — **and `refit_folds: True` for foundation/pre-trained TFMs**; see the "Foundation models: set `refit_folds=True`" note in `references/model_patterns.md`. From-scratch NNs omit it), plus `_more_tags()` (with `can_refit_full: True`). Do **not** declare a `can_estimate_memory_usage_static` tag: AutoGluon derives it from whether you implement `_estimate_memory_usage_static`. **Only torch models** (`AbstractTorchModel`) additionally implement `get_device()` / `_set_device()`; non-torch GPU models on `AbstractModel` must NOT (they have no `.to(device)`).
- Docstring must include: description, paper title, authors, codebase URL, license
- Keep optional third-party imports (the wrapped library itself) inside `_fit` / per-method scope so importing this module never requires the optional dep at top-level
- Decide the model's untimed **warm-up** (Step 3g) while you have the library docs in hand
- **Foundation model with Hugging Face weights: always pin `revision=`** on every `hf_hub_download` / `snapshot_download` call the wrapper or its `prefetch_weights` makes, and pass the pinned file to the library where it takes a path; never resolve against the repo's moving default branch. See "Foundation-model weights: always pin the HF checkpoint revision" in `references/model_patterns.md` for how to resolve the commit.

### 3c. `packages/tabarena/src/tabarena/models/{ModelKey}/hpo.py`

The search-space generator. By default use an **empty search space** (like TabPFN-2.6) — only add hyperparameters if the user explicitly asks or if the model has obvious tunable knobs. See template in `references/model_patterns.md` section "hpo.py template".

### 3d. `packages/tabarena/src/tabarena/models/{ModelKey}/info.py`

Defines `{ModelKey}_method_metadata: MethodMetadata` and `{ModelKey}_info: ModelInfo`. `info.py` is the single source the auto-discovery registry walks — populating it correctly is how the model becomes visible to `discover_models()`. See template in `references/model_patterns.md` section "info.py template".

**When you set `ag_key`/`model_key` here, also classify the model in `get_model_family` (Step 4d)** — those keys decide the model's leaderboard family, and skipping this makes it show as ❓ Other on the website.

### 3e. Multi-file support code (optional)

If the wrapper needs helper modules (preprocessors, vendored upstream code, large internal classes), put them in a private subfolder of `packages/tabarena/src/tabarena/models/{ModelKey}/`:

- `_internal/` — for hand-written helpers (preprocessors, internal classes, adapters)
- `_vendor/` — only for code copied verbatim from an upstream project; keep the original layout/license alongside

Both subfolders need their own empty `__init__.py`. Import them from `model.py` via absolute paths, e.g. `from tabarena.models.{ModelKey}._internal.preprocessing import Preprocessor`.

### 3f. Test config (no per-model test file)

There is **no per-model test file**. `tests/tabarena/models/test_all_models.py`
is parametrized over the model registry, so it fits the new model automatically once
its `info.py` is discoverable. It skips on `ImportError` (optional dep missing) and for
GPU-only models without CUDA.

Only touch `tests/tabarena/models/smoke_configs.py` if the model's toy fit needs
a speed-up: add one entry to `SMOKE_OVERRIDES`, keyed by the model's `MethodMetadata.method`
(the registry key), e.g. `"{ModelName}": ModelSmokeTest({"max_epochs": 1})`, or
`ModelSmokeTest(problem_types=("regression",))` for a regression-only model. If the model
fits fine with default hyperparameters on all problem types, add nothing. A wrapper that declares its
cheapness knobs as the `cheap_hyperparameters` ClassVar (Step 3g) needs no entry either: `smoke_for`
merges them into the smoke config and the warm-up dummy fit uses the same dict.

### 3g. Warm-up (untimed environment warm-up): decide, don't skip

TabArena runs an untimed warm-up before every timed fit (`AbstractExecModel.warmup_fn`, dispatched
per model class by `tabarena.models.warmup.warmup_model_cls`), so one-time per-environment costs
(library imports, JIT and kernel compilation, the CUDA context, pretrained weights) stay out of
`time_train_s` / `time_infer_s` and the fit time limit. The layers are additive and run in this
order for every model; a new model declares only what the generic layers cannot see:

1. An optional `warmup` classmethod, `warmup(cls, *, problem_type=None, num_cpus=None,
   num_gpus=None, hyperparameters=None, **kwargs) -> None`, for work the declarative layers cannot
   express (a library's own kernel pre-compilation, allocator settings that must precede the CUDA
   context). References: `chimeraboost`, `mitra_v2`.
2. Torch import plus CUDA context for `AbstractTorchModel` subclasses (automatic).
3. `warmup_modules: ClassVar[tuple[str, ...]] = ("yourlib", "yourlib.submodule")`: the modules
   `_fit` and `_predict` import lazily, merged over the MRO. A `"torch"` entry also creates the CUDA
   context for a torch-backed model on plain `AbstractModel`.
4. The `ag_key` map in `warmup.py` for AutoGluon built-ins (LightGBM, CatBoost, XGBoost, ...).
5. A dummy fit and predict on a small synthetic dataset (`make_synthetic_frames`, fixed seed, no
   task data), which triggers the lazy imports, library caches and kernel loads a first fit pays.
   For a class that declares `shared_weights` (below) it also builds and registers the network the
   timed fit reuses. Opt-outs on the class: `warmup_dummy_fit = False`, `warmup_dummy_fit_kwargs`
   (`n_rows`, `n_features`, `n_categorical`, `time_limit`) and `cheap_hyperparameters` (cheapness
   knobs such as `n_estimators=1`, merged over the config; never an input of the network loader,
   since the dummy fit must build the network the real fit looks up).

| Situation | Declare |
|---|---|
| sklearn-like or lightweight | Nothing; the dummy fit covers it. |
| Torch model on `AbstractTorchModel` | `warmup_modules` with the heavy extra imports (`transformers`, ...). |
| Torch-backed model on `AbstractModel` | `warmup_modules = ("torch", "yourlib")`. References: `modernnca`, `xrfm`, `tabstar`. |
| Library JIT-compiles kernels (numba, JAX, custom CUDA) | A `warmup` classmethod calling the library's pre-compile entry point when one exists (`chimeraboost`, whose `warmup()` needs `chimeraboost>=0.14.1`). Ask the user for the entry point and minimum version when the docs do not say. |
| Foundation model with pretrained weights | A `shared_weights` declaration (below) plus `warmup_modules` for the library. |

Fairness contract (the `tabarena/models/warmup.py` module docstring is the reference): data-independent
work only; never task data, never task- or data-specific state carried into the fit, never a global
random number generator advanced.

**Foundation models share one network.** A bagged fit of an in-context model would otherwise build
the same frozen network once per fold child and once more for the refit child, inside the timed fit.
Every such library builds its network inside one call its `fit` makes: a method on the estimator
(tabpfn `_initialize_model_variables`, tabicl `_load_model`), a module function (causilo
`load_pretrained_model`) or a classmethod (`Tab2D.from_pretrained`). The wrapper names that call and
the inputs that decide which network it builds; AutoGluon's `AbstractTorchModel` does the rest:

```python
from autogluon.core.models.abstract import SharedWeights

class {ClassName}Model(AbstractTorchModel):
    shared_weights: ClassVar[SharedWeights] = SharedWeights(
        loader=("somelib:SomeClassifier._load_model", "somelib:SomeRegressor._load_model"),
        key=("checkpoint_version", "model_path"),  # the loader's inputs that pick the network
        disabled_by=("kv_cache",),                  # inputs under which the library writes into it
    )
    cheap_hyperparameters: ClassVar[dict] = {"n_estimators": 1}
```

`loader` is `"package.module:Class.method"` for a method the estimator's `fit` calls, or
`"package.module:function"` (also `Class.classmethod`) for a call that returns the network; several
when the library has one class per task. `key` names the estimator attributes (method) or call
arguments (function) that decide which network is built; the device is part of the key by itself,
from the loader's `device` input or, for a loader without one, from `self.device` when `_fit` sets it
before the fit, else the fit's `num_gpus`. `disabled_by` lists inputs (names, or a predicate over the
inputs) under which the library writes into or casts the network, so that fit builds its own.
`copy_per_fit=True` is for a fine-tuning model: every fit trains a deep copy and pickles it. Nothing in
`_fit` changes: the library's own code runs on the first call in the process and registers what it
produced; every later call with the same inputs and device reuses it, the fitted model pickles without
the network and takes it back on load, `set_device` swaps registry entries, and
`get_info()["shared_weights"]` records it. Sharing is off for a fit with
`ag_args_fit={"share_pretrained_weights": False}`, per class through
`model_class_settings={"<ag_key>": {"share_weights": False}}`, and for a configuration named in
`disabled_by`; fold fitting and `refit_folds` stay the wrapper's ordinary ensemble arguments.

**The library contract.** What a library has to offer is one loader call whose inputs are estimator
parameters (checkpoint name or path, device, and any flag that shapes the network) and whose effect
is to produce the network, as a return value or as attributes it sets. Check this before integrating:
if the library loads inside `__init__` (TabDPT, EXAONE Tabular), ask the maintainers for a separable
loader or a `network=` constructor argument first, and only then fall back to an adapter in
`<model>/_estimators.py`: a `load_network(...)` function that replicates the loading half of the
constructor (the declared loader) and, when the constructor cannot take a prebuilt network, a
constructor replica. Mark the module as a developer fix in its docstring, say what the library should
offer instead, and guard the replica against library bumps (`tabdpt/_estimators.py`,
`exaone_tabular/_estimators.py`). A library whose `__setstate__` re-runs its loader (tabicl) reloads
shared for free; one that pickles its network (tabpfn) is covered by the generic weightless pickle.
Wrappers that do not care about sharing simply declare nothing (TabDPT v1.1, TabPFN-Wide, SAP-RPT-OSS,
iLTM whose library caches for itself).

`references/model_patterns.md` under "Shared pretrained weights" has the field-by-field reference;
`tabicl/model.py` is the plain in-tree case, `causilo/model.py` a function loader without a device
input, `mitra_v2/model.py` a fine-tuning model, `tabdpt/` and `exaone_tabular/` the adapters.

Tests: `tests/tabarena/models/test_shared_weights_models.py` discovers every registered class that
declares `shared_weights` and checks the declaration (well-formed, the loader resolves when the library
is installed, the keyed inputs are the loader's parameters, the named disablers switch sharing off);
nothing to register per wrapper. `pytest -m models tests/tabarena/models/test_shared_weights_models.py
-k <Model>` fits the real library twice, asserts one network, a weightless pickle that reloads and
predicts the same, and equal predictions with sharing switched off.

**Inference side.** The exec model persists the fitted model in memory around the predict timer
(`AGWrapper.persist`, through `persist_inference.persist_for_inference`) and calls an optional
instance method `prepare_for_inference(self) -> None` on every persisted object, bagged children
included. Its contract is written once, in `AbstractExecModel.pre_predict` (`exec_models/base.py`);
refer to it instead of restating it. Data-dependent first-predict work (`torch.compile` on the real
batch, kernel dispatch for the real shapes) is measured by design.

**Check it.** `python -P -m tabarena.tools.audit_warmup --model <Method>` (from the repo root) runs
the warm-up, then a fit and predict on synthetic data, and prints the warm-up report and the packages
each timed section still imported. `pytest -m models tests/tabarena/models/test_warmup_coverage.py -k <Method>`
asserts the same per registered model; `allowed_lazy_imports` on `ModelSmokeTest` is the escape
hatch and needs a comment. Both see the main process only: parallel Ray fold workers start cold, and
code vendored under `tabarena.models` is invisible to the diff. Report the warm-up decision and this
limitation in Step 8.

## Step 4: Edit existing files

Edit both locations **in a single pass** (read each file first, then edit):

### 4a. `packages/tabarena/src/tabarena/models/__init__.py`

Add a lazy entry for the new class so `from tabarena.models import {ClassName}Model` works:
```python
_LAZY_CLASSES = {
    ...
    "{ClassName}Model": "tabarena.models.{ModelKey}.model",
    ...
}
```
Also add `"{ClassName}Model"` to `__all__` and (under `TYPE_CHECKING`) to the static `from tabarena.models.{ModelKey}.model import {ClassName}Model` block, both kept alphabetised.

### 4b. `packages/tabarena/src/tabarena/models/utils.py` — no edit needed

**No longer required** (auto-registry). `get_configs_generator_from_name()` now resolves the
search space from the auto-discovered `MODEL_REGISTRY` (via `get_model_info_from_name`); there is
no `name_to_import_map` dict to update. Creating `info.py` (Step 3d) is what makes the model
resolvable by name. Leave `utils.py` untouched.

### 4c. `packages/tabarena/pyproject.toml`

The `pyproject.toml` defines a per-model extra for every supported model, plus three union extras built via **self-references** (`"tabarena[<name>]"`):

- **`benchmark`** — the curated core set used for standard benchmarking. Stable and resolver-friendly. Do **not** add a new model here unless the user explicitly says it belongs in the core set.
- **`extended`** — the layered set installed on top of `benchmark` for the broader model zoo. **This is where most new models go.**
- **`all`** — experimental union of `benchmark` + `extended` + special-cased extras like `probmetrics` (which has conflict-prone deps and is excluded from `extended` on purpose). Updated automatically via `tabarena[extended]`, so usually no manual edit needed unless the model is conflict-prone.

Always declare the pip spec exactly once in the per-model extra, then reference the model by name in the union(s). Never paste the raw `{pip_package}` into a union extra.

**Step 1 — declare the per-model extra** under `[project.optional-dependencies]`:
```toml
{ModelKey} = ["{pip_package}"]
```

**Step 2 — add it to the right union via self-reference**:

| Situation | Edit |
|---|---|
| **Default**: new extended model | Add `"tabarena[{ModelKey}]"` to the `extended` extra. |
| Core benchmark model (only if user explicitly says so) | Add `"tabarena[{ModelKey}]"` to the `benchmark` extra. |
| Model has known dependency conflicts (rare, like `probmetrics`) | Skip both `benchmark` and `extended`; add `"tabarena[{ModelKey}]"` to `all` only. |
| We reuse only a slice of a heavy library's code (`--no-deps` pattern, like `denselight` reusing LightAutoML's NN stack) | Declare the per-model extra, but leave it OUT of `benchmark`/`extended`/`all` and document the `--no-deps` install (see below). |

After this, users can install the model alone (`uv sync --extra benchmark --extra {ModelKey}`), as part of the extended set (`uv sync --extra benchmark --extra extended`), or via `--extra all`.

**`--no-deps` minimal-reuse pattern.** Sometimes the goal is to reuse only a small part of a large
library (e.g. one model class + its training loop) without pulling that library's heavy,
conflict-prone transitive tree. If the *specific* modules you import resolve using only packages
TabArena already ships (`torch`, `numpy`, `pandas`, `scikit-learn`, `catboost`, `lightgbm`,
`xgboost`, `tqdm`), then the library can be installed with `pip install <lib> --no-deps`. To check:
clone the upstream repo and trace the import chain of *exactly* the modules your wrapper imports
(package `__init__.py`s that only declare `__all__` without eager submodule imports are the enabler —
they let you reach a deep submodule without triggering the whole package). The pip extra (`{ModelKey}
= ["<lib>"]`) still names the library so the drift checker passes, but a plain `pip install
tabarena[{ModelKey}]` would pull the full tree — so keep it out of the union extras and document
the `--no-deps` install in the wrapper docstring + the pyproject comment. `denselight` is the
reference example. Keep all `<lib>` imports lazy (inside `_fit` / a private `_internal/` runner)
so model discovery never needs the optional dep.

**Step 3 — verify the per-model extra matches `info.py`** with the drift checker:

```bash
python -m tabarena.tools.sync_pyproject_extras
```

`packages/tabarena/src/tabarena/tools/sync_pyproject_extras.py` aggregates every `ModelInfo.pip_extra` from the registry and compares it against `[project.optional-dependencies]` in `packages/tabarena/pyproject.toml`, printing per-folder `OK`/`DRIFT`. Add `--check` to make it exit non-zero on drift (CI mode). Run it after editing either side so the two stay in sync.

### 4d. `packages/tabarena/src/tabarena/website/website_format.py` — leaderboard model family

`get_model_family()` decides the model's leaderboard **Type** (emoji) and **TypeName** — one of `Tree-based`, `Foundation Model`, `Neural Network`, `Baseline`, `Reference Pipeline`, or `Other`. It prefix-matches the model's `config_type` against `prefixes_mapping`. A model's `config_type` is its **`model_key`** — which defaults to `ag_key` only when `model_key` is left unset, so the two can differ: e.g. TabFM sets `model_key="TABFM"` with `ag_key="TA-TABFM"`, so its `config_type` is `TABFM`. Add the **uppercased `config_type`** to the correct family list (a leading `TA-` is stripped before matching, so the `TA-`/bare forms are equivalent):

```python
prefixes_mapping = {
    Constants.tree:           [..., "{config_type_upper}"],   # tree-based boosters/forests
    Constants.foundational:   [..., "{config_type_upper}"],   # pre-trained / in-context-learning models
    Constants.neural_network: [..., "{config_type_upper}"],   # NNs trained from scratch
    ...
}
```

If you skip this, the model falls through to `Other` (❓) on the leaderboard. The effect is only visible once the model appears on the website, but it's cheapest to set here while you have the `ag_key` in hand. Matching is by **prefix**, so when one `ag_key` is a prefix of another (e.g. `TABICL` vs `TABICLV2`), list both and keep them in the intended family. This drives only the `Type`/`TypeName` columns of the generated `website_leaderboard.csv` — the plots use a separate grouping — so an already-generated leaderboard can be re-patched in place rather than fully regenerated.

## Step 5: Auto-derived registries (no manual edit)

These pieces pick up the new model automatically once Step 3 lands — do not edit them by hand:

- `packages/tabarena/src/tabarena/models/_registry.py` — `discover_models()` walks `tabarena/models/*/info.py` and collects every `ModelInfo` found. As long as `info.py` exports a top-level `ModelInfo` instance, the model joins `MODEL_REGISTRY`.
- `packages/tabarena/src/tabarena/benchmark/exec_models/registry.py` — auto-derives `tabarena_model_registry` from `get_model_registry()`, so the new class becomes available through the AG registry on the next import.

## Step 6: Lint

Run ruff on the new files (add `tests/tabarena/models/smoke_configs.py` only if you edited it):
```bash
ruff check --fix packages/tabarena/src/tabarena/models/{ModelKey}/
```

Fix any reported issues.

## Step 7: Arena registration (optional — only if the model has been benchmarked)

If the model already has processed + uploaded benchmark results, register it in the arena collection so it appears in the benchmark. For the default `tabarena` arena, edit `packages/tabarena/src/tabarena/contexts/tabarena/methods.py` (read it first):

1. **Import** the `{model_key}_method_metadata` you defined in `info.py` in the alphabetical per-model import block (recent additions use the plain name, e.g. `from tabarena.models.nori.info import nori_method_metadata`).
2. **Add the entry** to `tabarena_method_metadata_collection.method_metadata_lst`, under the matching group comment (CPU vs neural/GPU/foundation models).

This mirrors the **`upload-method`** skill's registration step — use that skill for the full process-and-upload flow (the entry only resolves to downloadable artifacts once the results are actually uploaded).

**If the model has not been benchmarked yet, skip this step entirely** — `info.py` already declares the metadata for the registry; the collection entry is only needed when results artifacts actually exist.

## Step 8: Report

Summarize what was created/edited:
- List new files created
- List files edited and what was added
- The warm-up decision (Step 3g): what is warmed and why (or why nothing is needed), any open
  question for the user (library warm-up entry point / version), and the worker-process limitation
- Note any TODOs left for the user (e.g., implementing `_predict_proba` if the library API is unclear, tuning `ag_priority`, adding a real search space later, registering benchmark artifacts after a real run)

When asked to open the PR, use `.github/pull_request_template.md`: a two-to-four sentence summary,
everything longer inside the collapsed `<details><summary>Details</summary>` block, the commands run
under Tests. Fill in the "Model or system submission" section for a model (delete the system lines)
and keep the closing contribution line. Do not paste this Report into the PR body; the Report is for
the chat, the PR body is for the reviewers. State the TabArena-Lite (or BeyondArena `core`) results
with the hardware and the entry-point script if they exist; if they do not, say so, since a
maintainer will ask (TabArena verifies submitted results by re-running them, it does not benchmark
on request). Questions go through the issue forms in `.github/ISSUE_TEMPLATE/`.
