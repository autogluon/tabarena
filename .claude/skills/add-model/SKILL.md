---
name: add-model
description: Add a new ML model to the TabArena benchmark system. Use this skill whenever the user wants to integrate a new tabular ML model into TabArena — even if they just say "add X model", "integrate X", "support X", or "wrap X for the benchmark". Creates all required files: the AutoGluon model wrapper, the search space generator, registry entries, pyproject.toml dependency, and a test file. Reads existing similar models for inspiration and optionally fetches documentation URLs to understand the new model's API.
argument-hint: <ModelName> [<pip-package>] [<doc-url>]
user-invocable: true
---

# Add Model to TabArena

This skill integrates a new tabular ML model into the TabArena benchmark. It creates or edits **7 locations** across the codebase, plus an optional metadata artifact entry.

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
| Foundation / pre-trained / GPU (e.g. TabPFN, SAP-RPT-OSS, TabSTAR) | `AbstractTorchModel` | `tabarena/tabarena/benchmark/models/ag/sap_rpt_oss/sap_rpt_oss_model.py` |
| Torch NN trained from scratch (e.g. TabM, RealMLP) | `AbstractTorchModel` | `tabarena/tabarena/benchmark/models/ag/tabm/tabm_model.py` |
| CPU / sklearn-like (e.g. EBM, KNN) | `AbstractModel` | `tabarena/tabarena/benchmark/models/ag/ebm/ebm_model.py` |

Read the reference model file now (use the Read tool). Use it as a structural guide — you will adapt rather than copy.

Also read the search space reference: `references/model_patterns.md` in this skill's directory — it contains annotated code patterns for both `_model.py` and `generate.py`.

## Step 3: Create new files

Create these files (exact paths relative to the repo root `/work/dlclarge2/purucker-tabarena/code/tabarena_new/tabarena/`):

### 3a. `tabarena/tabarena/benchmark/models/ag/{ModelKey}/__init__.py`
```python
from __future__ import annotations
```

### 3b. `tabarena/tabarena/benchmark/models/ag/{ModelKey}/{ModelKey}_model.py`

Use the template in `references/model_patterns.md` section "Model wrapper template". Key points:
- Start with `from __future__ import annotations`
- Inherit from `AbstractTorchModel` (GPU/torch models) or `AbstractModel` (CPU models)
- Set `ag_key`, `ag_name`, `ag_priority = 65`, `seed_name = "random_state"`
- Implement `_fit()`, `_set_default_params()`, `supported_problem_types()`
- For GPU models: also implement `get_device()`, `_set_device()`, `_get_default_resources()`, `get_minimum_resources()`, `_get_default_ag_args_ensemble()` (with `fold_fitting_strategy: sequential_local`), `_class_tags()` (with `can_estimate_memory_usage_static: False`), `_more_tags()` (with `can_refit_full: True`)
- Docstring must include: description, paper title, authors, codebase URL, license

### 3c. `tabarena/tabarena/models/{ModelKey}/__init__.py`
```python
from __future__ import annotations
```

### 3d. `tabarena/tabarena/models/{ModelKey}/generate.py`

By default use an **empty search space** (like TabPFN-2.6). Only add hyperparameters if the user explicitly asks or if the model has obvious tunable knobs. See template in `references/model_patterns.md` section "generate.py template".

### 3e. `tst/benchmark/models/test_{ModelKey}.py`

See template in `references/model_patterns.md` section "Test template". Include a minimal `FitHelper.verify_model()` call with `model_hyperparameters={}` (add a speed-up param if the model has one like `max_epochs=1`).

## Step 4: Edit existing files

Edit all 4 locations **in a single pass** (read each file first, then edit):

### 4a. `tabarena/tabarena/benchmark/models/ag/__init__.py`
Add import line (alphabetically sorted by class name):
```python
from tabarena.benchmark.models.ag.{ModelKey}.{ModelKey}_model import {ClassName}Model
```
Add `"{ClassName}Model"` to `__all__` (keep list sorted).

### 4b. `tabarena/tabarena/benchmark/models/model_registry.py`
Add to the import block and to `_models_to_add` list.

### 4c. `tabarena/tabarena/models/utils.py`
Add to `name_to_import_map` dict:
```python
"{ag_name}": lambda: importlib.import_module("tabarena.models.{ModelKey}.generate").gen_{ModelKey},
```
The key must be the `ag_name` string (e.g. `"TA-TabPFN-2.6"`).

### 4d. `tabarena/pyproject.toml`

The `pyproject.toml` defines a per-model extra for every supported model, plus three union extras built via **self-references** (`"tabarena[<name>]"`):

- **`benchmark`** — the curated core set used for standard benchmarking (currently `tabpfn`, `tabicl`, `ebm`, `search_spaces`, `realmlp`, `tabdpt`, `tabm`). Stable and resolver-friendly. Do **not** add a new model here unless the user explicitly says it belongs in the core set.
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

After this, users can install the model alone (`uv sync --extra benchmark --extra {ModelKey}`), as part of the extended set (`uv sync --extra benchmark --extra extended`), or via `--extra all`.

## Step 5: Lint

Run ruff on the new files:
```bash
cd /work/dlclarge2/purucker-tabarena/code/tabarena_new/tabarena
ruff check --fix tabarena/tabarena/benchmark/models/ag/{ModelKey}/ tabarena/tabarena/models/{ModelKey}/ tst/benchmark/models/test_{ModelKey}.py
```

Fix any reported issues.

## Step 6: Metadata artifact (optional — only if the model has been benchmarked)

If the model already has benchmark results to register in TabArena's artifact system, add a metadata entry. Check the most recent metadata file to see the pattern:

```
tabarena/tabarena/nips2025_utils/artifacts/_tabarena_method_metadata_YYYY_MM_DD.py
```

The naming convention is `_tabarena_method_metadata_YYYY_MM_DD.py` where the date reflects when this batch of results was produced. Either add to the latest file or create a new dated file if the benchmarking run is new.

Each entry is a `MethodMetadata(...)` object. Example:

```python
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

{ModelKey}_metadata = MethodMetadata(
    method="{ModelName}",           # e.g. "TabSTAR"
    method_type="config",
    display_name="{ModelName}",
    compute="gpu",                  # or "cpu"
    date="YYYY-MM-DD",             # date results were generated
    ag_key="{ag_key_without_TA}",  # e.g. "TABSTAR" (matches TabSTARModel.ag_key)
    model_key="{ag_key_without_TA}",
    config_default="{ModelName}_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-YYYY-MM-DD",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    verified=True,
    reference_url="{doc_url}",
    cache_type="r2",
)
```

Then import and expose it in `_tabarena_method_metadata.py`:
```python
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_YYYY_MM_DD import (
    {ModelKey}_metadata,
)
```

**If the model has not been benchmarked yet, skip this step entirely.**

## Step 7: Report

Summarize what was created/edited:
- List new files created
- List files edited and what was added
- Note any TODOs left for the user (e.g., implementing `_predict_proba` if the library API is unclear, tuning `ag_priority`, adding a real search space later, adding metadata after benchmarking)
