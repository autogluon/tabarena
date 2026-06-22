---
name: add-model
description: Add a new ML model to the TabArena benchmark system. Use this skill whenever the user wants to integrate a new tabular ML model into TabArena — even if they just say "add X model", "integrate X", "support X", or "wrap X for the benchmark". Creates all required files: the AutoGluon model wrapper, the search-space generator, the per-model `info.py`, and the `pyproject.toml` extra (the model is fit-tested automatically by the registry-driven `test_all_models.py` — no per-model test file). Reads existing similar models for inspiration and optionally fetches documentation URLs to understand the new model's API.
argument-hint: <ModelName> [<pip-package>] [<doc-url>]
user-invocable: true
---

# Add Model to TabArena

This skill integrates a new tabular ML model into the TabArena benchmark.

Every model lives in **one folder** at `packages/tabarena/src/tabarena/models/<ModelKey>/`. That folder contains the wrapper, the HPO generator, and the metadata — and is auto-discovered by `tabarena.models._registry.discover_models()`. There is no separate `benchmark/models/ag/` layout anymore.

Per model, you create up to 5 source files, then edit two existing files. There is no per-model test file — the model is fit-tested automatically by the registry-driven `tests/tabarena/models/test_all_models.py`.

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
| Foundation / pre-trained / GPU (e.g. TabPFN, SAP-RPT-OSS, TabSTAR) | `AbstractTorchModel` | `packages/tabarena/src/tabarena/models/sap_rpt_oss/model.py` |
| Torch NN trained from scratch (e.g. TabM, RealMLP) | `AbstractTorchModel` | `packages/tabarena/src/tabarena/models/tabm/model.py` |
| CPU / sklearn-like (e.g. KNN) | `AbstractModel` | `packages/tabarena/src/tabarena/models/knn/model.py` |

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
- Inherit from `AbstractTorchModel` (GPU/torch models) or `AbstractModel` (CPU models)
- Set `ag_key`, `ag_name`, `ag_priority = 65`, `seed_name = "random_state"`
- Implement `_fit()`, `_set_default_params()`, `supported_problem_types()`
- **Honor the `_fit` contract** (read `references/model_patterns.md` → "The `_fit` contract"). The most common review findings on new wrappers are: ignoring the provided `X_val`/`y_val` (and instead auto-splitting a second holdout), ignoring `time_limit`, hardcoding the thread count instead of wiring `num_cpus`, and label-encoding + `fillna(0)` categoricals when the library handles them natively. `models/realmlp/model.py` is the reference for all of these.
- For GPU models: also implement `get_device()`, `_set_device()`, `_get_default_resources()`, `get_minimum_resources()`, `_get_default_ag_args_ensemble()` (with `fold_fitting_strategy: sequential_local`), `_class_tags()` (with `can_estimate_memory_usage_static: False`), `_more_tags()` (with `can_refit_full: True`)
- Docstring must include: description, paper title, authors, codebase URL, license
- Keep optional third-party imports (the wrapped library itself) inside `_fit` / per-method scope so importing this module never requires the optional dep at top-level

### 3c. `packages/tabarena/src/tabarena/models/{ModelKey}/hpo.py`

The search-space generator. By default use an **empty search space** (like TabPFN-2.6) — only add hyperparameters if the user explicitly asks or if the model has obvious tunable knobs. See template in `references/model_patterns.md` section "hpo.py template".

### 3d. `packages/tabarena/src/tabarena/models/{ModelKey}/info.py`

Defines `{ModelKey}_method_metadata: MethodMetadata` and `{ModelKey}_info: ModelInfo`. `info.py` is the single source the auto-discovery registry walks — populating it correctly is how the model becomes visible to `discover_models()`. See template in `references/model_patterns.md` section "info.py template".

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
fits fine with default hyperparameters on all problem types, add nothing.

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

## Step 7: Metadata artifact (optional — only if the model has been benchmarked)

If the model already has benchmark results to register in TabArena's artifact system, add a metadata entry to the dated batch file:

```
packages/tabarena/src/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_YYYY_MM_DD.py
```

Either add to the latest file or create a new dated file if the benchmarking run is new.

Each entry is a `MethodMetadata(...)` object (same class used in `info.py`, so the entry can be the `{ModelKey}_method_metadata` you already defined). Then import it in `_tabarena_method_metadata.py`:
```python
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_YYYY_MM_DD import (
    {ModelKey}_metadata,
)
```

**If the model has not been benchmarked yet, skip this step entirely** — `info.py` already declares the metadata for the registry; the artifact entry is only needed when results files actually exist.

## Step 8: Report

Summarize what was created/edited:
- List new files created
- List files edited and what was added
- Note any TODOs left for the user (e.g., implementing `_predict_proba` if the library API is unclear, tuning `ag_priority`, adding a real search space later, registering benchmark artifacts after a real run)
