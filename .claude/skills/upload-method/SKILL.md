---
name: upload-method
description: Process, upload, and register one benchmarked method's results in TabArena. Use this skill whenever a maintainer points at a benchmark run's output directory and wants to turn its raw `results.pkl` files into cached + hosted + registered TabArena artifacts — e.g. "upload this method", "process and upload these results", "host/publish <model>'s results", "register <model> in the leaderboard". By default Claude runs the whole flow itself (inspect → fix `info.py` → process → upload dry-run → real r2 upload → register in `methods.py`) after first telling the maintainer what it will do; it hands a step off via a command sheet only when the environment can't run it (no `tabarena[benchmark]` venv, no R2 credentials).
argument-hint: <run-data-dir> [<model>]
user-invocable: true
---

# Process & Upload a Benchmarked Method to TabArena

This skill drives the maintainer workflow that turns a benchmark run's **already-present** raw
`results.pkl` files into cached, hosted (r2), and registered TabArena artifacts. There is **no
download and no auto-generation** — the raw data is assumed to be on disk already (e.g. you unzipped
a submission, or it's a fresh run's `output/<run>/data` dir).

The authoritative prose for this flow is **AGENTS.md → "Processing & uploading method artifacts
(maintainers)"** and the two scripts' module docstrings (`scripts/run_process_method.py`,
`scripts/run_upload_results.py`). This skill operationalizes it: **Claude executes the flow itself
by default**, and makes the small deterministic code edits along the way.

## What this skill delivers

1. **A plan, stated first**: which methods, which run dir, which suite, and that the flow ends in a
   real r2 upload — tell the maintainer before executing (no need to wait for approval unless they
   asked to review something first).
2. **Claude-run execution**, in order per method:
   - `inspect` — read the raw data, print inferred fields + a metadata diff.
   - `process` — build + cache `metadata.yaml` + `processed/` + `results/` locally.
   - `upload` (dry-run, then `--no-dry-run`) — push the cached artifacts to r2.
3. **Code edits**:
   - The model's `info.py` `MethodMetadata` — fill in the manual upload fields, fix any
     raw-data mismatches the inspect diff surfaces.
   - The arena collection registration in `methods.py` so the method appears in the benchmark.
4. **A fallback command sheet** only for steps the environment can't run.

Execution requirements (check before starting; hand off the affected step if missing):
- a venv with **`tabarena[benchmark]`** installed (look under `~/.venvs/tabarena_*`; run scripts with
  its python from the repo root),
- **R2 credentials in the environment** for the real upload (`printenv | grep -o '^R2_[A-Z_]*'`
  should show `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY`; never pass them as flags).

Processing and uploads are long-running: run them **in the background** (per-method), watch with a
monitor (per-method DONE/FAILED lines + a stall watchdog), and for multi-method batches pipeline the
uploads — start each method's dry-run + real upload as soon as its processing finishes.

## Step 0: Gather inputs

Parse `$ARGUMENTS`. Collect (ask only for what's missing or ambiguous):

| Input | Example | Notes |
|---|---|---|
| `run_data_dir` | `.../output/benchmark_chimeraboost_16062026/data` | Dir of raw `results.pkl` files (searched recursively). Point at the run's `data/`. |
| `model` | `chimeraboost` | The `packages/tabarena/src/tabarena/models/<model>/` folder. Usually inferable from the run/data dir name; confirm. |
| `suite` | `tabarena-2026-06-30` | The dated run/suite id. **Must differ from `method`** (see Step 3). Default to `tabarena-<run-date>`; ask if unclear. |
| `arena` | `tabarena` | Which arena collection to register in (default `tabarena`; e.g. `beyondarena` has its own). |
| `verified` | `False` until signed off | Whether the results are verified. Default keep `False`; flip to `True` only when the maintainer confirms (Step 3). |

## Step 1: Locate the model's `MethodMetadata`

Read `packages/tabarena/src/tabarena/models/<model>/info.py` and note the **exact** variable name of
its `MethodMetadata` (e.g. `chimeraboost_method_metadata`, `nori_method_metadata`). Both scripts take
it as a dotted reference:

```
tabarena.models.<model>.info:<varname>
```

- **`info.py` exists** (the normal case — the model was added via the `add-model` skill): it already
  carries the raw-inferable fields (`ag_key`, `config_default`, `can_hpo`, `is_bag`, `compute`,
  `method_type`). Claude only fills the *manual* upload fields in Step 3.
- **No `info.py`** (a raw external submission): the model isn't integrated yet. Run `inspect`
  (Step 2) to get the copy-paste `MethodMetadata.<type>(...)` snippet, then author
  `models/<model>/info.py` from it (use the **`add-model`** skill if the model also needs a wrapper).
  Processing **requires** an explicit committed `MethodMetadata` — it refuses to guess.

## Step 2: Inspect the raw data (Claude runs; confirms inference)

```bash
<venv>/bin/python scripts/run_process_method.py <run_data_dir>
```

This prints the fields inferred from the raw data (`method_type`, `ag_key`, `compute`,
`config_default`, `can_hpo`, `is_bag`, task/problem-type/metric coverage), a suggested
`MethodMetadata` snippet, and — when an `info.py` metadata is passed — an **inferred-vs-provided
diff**. Use it to confirm `info.py` matches the raw data before processing. Two gotchas it catches:

- **`config_default` is compared post-rename**: configs are renamed to the method's prefix during
  processing, so a `config_default` authored with the raw prefix won't match. The snippet/diff shows
  the post-rename value — use that.
- **`method != suite`**: `process` fails if they're equal (suite defaults to method when unset).

**Multi-method run dirs**: if the run's `data/` holds several methods' config dirs side by side,
`run_process_method.py` can't split them — write a small gitignored driver in `tmp_scripts/` that
discovers the `results.pkl` paths once, splits them by top-level config-dir prefix, and calls
`_infer_from_raw` / `verify_method_metadata` / `process_raw(file_paths=...)` per method (see the
`multi-method-run-upload` pattern; a `--c1-only` pass over just the `_c1_` dirs makes
`config_default` inferable even for HPO methods with hundreds of configs).

## Step 3: Edit `info.py` (Claude does now)

Fill in the **manual** fields the upload requires. These are not inferable from raw data, so the
maintainer normally hand-edits them — Claude does it now. Read the file first, then `Edit`:

| Field | Set to | Why |
|---|---|---|
| `suite` | the dated run id, e.g. `"tabarena-2026-06-30"` | **Required, must differ from `method`.** Equal pair fails `process`'s `method != suite` check (suite defaults to method when unset). |
| `cache_type` | `"r2"` | **Required for upload** — the upload script is r2-only; `"local"`/`None` has no remote store. |
| `cache_kwargs` | `{"bucket": "tabarena", "prefix": "cache"}` | The r2 location (`bucket` + `prefix`). Required when `cache_type="r2"`. |
| `date` | `"YYYY-MM-DD"` | The run date. Validated as a real calendar date. |
| `verified` | `False` (until signed off), then `True` | Manual trust flag. Keep `False` until the results are verified; flip to `True` once they are (typically the final step). |

Leave the raw-inferable fields (`ag_key`, `config_default`, `can_hpo`, `is_bag`, `compute`,
`method_type`) **as they are** — they came from `add-model`. Only change one if Step 2's diff shows a
genuine mismatch (and then to the inferred value).

Example — making `chimeraboost` upload-ready (the upload fields added; compare `nori`'s `info.py`,
which is already in this shape):

```python
chimeraboost_method_metadata = MethodMetadata.config(
    method="ChimeraBoost",
    suite="tabarena-2026-06-30",                              # added: distinct dated suite
    ag_key="CHIMERA",
    config_default="ChimeraBoost_c1_BAG_L1",
    compute="cpu",
    is_bag=False,
    date="2026-06-15",
    reference_url="https://github.com/bbstats/chimeraboost",
    display_name="ChimeraBoost",
    verified=False,                                           # flip to True once verified
    cache_type="r2",                                          # added
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},   # added
)
```

## Step 4: Execute process → upload (Claude runs, in the background)

Run these with the resolved dotted reference, per method — background + monitor for the slow ones.
The same block doubles as the fallback command sheet if a step must be handed to the maintainer.
Using `chimeraboost` as the example:

```bash
# 1. Inspect (confirm inferred fields match info.py) — already done in Step 2

# 2. (info.py edited: suite + cache_type/cache_kwargs + date, plus any inspect-diff fixes)

# 3. Process: build + cache metadata.yaml + processed/ + results/ locally
<venv>/bin/python scripts/run_process_method.py <run_data_dir> \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata --process

# 4. Upload dry-run: verifies every part exists locally and prints what/where (no creds needed)
<venv>/bin/python scripts/run_upload_results.py \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata

# 5. Real upload (R2 creds via env, NEVER flags):
<venv>/bin/python scripts/run_upload_results.py \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata --no-dry-run
```

After the real upload, verify on the bucket (ListObjects via boto3 against the R2 endpoint): each
method should have the full object set under
`cache/artifacts/<suite>/methods/<Method>/` (`metadata.yaml`, `processed.zip`,
`processed/configs_hyperparameters.json`, `raw.zip`, `results/*.parquet`).

Notes:
- `process` requires the explicit `--method-metadata` and verifies it against the raw data first; a
  real mismatch errors (override with `--ignore-metadata-mismatch`, a `method`-name mismatch only
  warns). Raw + HPO trajectories are cached by default (`--no-cache-raw` / `--no-cache-hpo-trajectories`).
- The dry-run is the default for the upload script; it needs no credentials and prints the exact
  `--no-dry-run` command. Raw uploads by default (`--no-upload-raw` to skip).
- For the real upload, set `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` in the
  environment (export them or use a `.env`) — never as CLI flags (they'd leak into shell history /
  the process table). The dry-run prints how to obtain them if unset
  (`MethodMetadata.r2_credentials_help()`).

## Step 5: Register in the arena collection (Claude does now)

Add the method to the collection so it appears in the benchmark. For the default `tabarena` arena,
edit `packages/tabarena/src/tabarena/contexts/tabarena/methods.py` (read it first):

1. **Import** the model's `info.py` metadata in the alphabetical import block, matching the file's
   existing style (most recent additions use the plain name, e.g.
   `from tabarena.models.nori.info import nori_method_metadata`; alias only on a name collision).
2. **Add the entry** to `tabarena_method_metadata_collection.method_metadata_lst`, under the matching
   group comment (`# Default tabular models (CPU)` vs `# Neural / GPU / foundation models`), placed by
   compute/type.

It flows into `tabarena_method_metadata_complete_collection` automatically (no separate edit). Other
arenas (e.g. `beyondarena`) register in their own collection's `methods.py`.

**Caveat to surface in the report:** the collection entry only resolves to *downloadable* artifacts
once the real upload (Step 4.5) has actually run. The code edit is safe to land now, but the method
won't load for others until uploaded.

## Step 6: Lint touched files

```bash
ruff check <touched-files>
ruff format --check <touched-files>
```

Touched files are the model's `info.py` and `contexts/<arena>/methods.py`. Fix anything reported
(the `from __future__ import annotations` import is already present in both — don't drop it).

## Step 7: Report

Tell the maintainer:

- **What was executed and verified**: inspect diff results, processing outcome per method, and the
  r2 destinations confirmed after the real upload — plus any data-quality warnings from processing
  (e.g. `Not close TEST` prediction-fidelity lines, with affected datasets and severity).
- **Edits Claude made**: the `info.py` upload fields (suite / cache_type / cache_kwargs / date /
  verified, plus any inspect-diff fixes) and the `methods.py` import + collection entry.
- **Steps handed off** (only if the env couldn't run one): the exact command(s) from Step 4.
- **Open decisions / TODOs**: whether to flip `verified` to `True` (only after sign-off), committing
  the working-tree edits, and — if the method should appear on the website — that `update-leaderboard`
  is the next lifecycle step.
