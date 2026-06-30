---
name: upload-method
description: Process, upload, and register one benchmarked method's results in TabArena. Use this skill whenever a maintainer points at a benchmark run's output directory and wants to turn its raw `results.pkl` files into cached + hosted + registered TabArena artifacts — e.g. "upload this method", "process and upload these results", "host/publish <model>'s results", "register <model> in the leaderboard". Produces the exact command sheet the maintainer runs (inspect → process → upload dry-run → upload), and edits the things Claude can do right away: the model's `info.py` `MethodMetadata` (suite / date / cache_type / cache_kwargs / verified) and the arena collection registration in `methods.py`.
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
`scripts/run_upload_results.py`). This skill operationalizes it: split the work into *what the
maintainer runs* (the slow, data-touching, credentialed commands) and *what Claude does right away*
(the small, deterministic code edits).

## What this skill delivers

1. **A command sheet** the maintainer copy-pastes and runs, in order:
   - `inspect` — read the raw data, print inferred fields + a metadata diff.
   - `process` — build + cache `metadata.yaml` + `processed/` + `results/` locally.
   - `upload` (dry-run, then `--no-dry-run`) — push the cached artifacts to r2.
2. **Edits Claude makes now** (no run needed):
   - The model's `info.py` `MethodMetadata` — fill in the manual upload fields.
   - The arena collection registration in `methods.py` so the method appears in the benchmark.

The maintainer runs the commands because they (a) load the raw data with `ray` and need
`tabarena[benchmark]` installed, (b) touch their local cache, and (c) need r2 credentials in the
environment for the real upload. Claude *may* run `inspect` itself if the env is available and it
helps confirm field values, but never the credentialed upload.

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

## Step 2: Inspect the raw data (maintainer runs; confirms inference)

```bash
python scripts/run_process_method.py <run_data_dir>
```

This prints the fields inferred from the raw data (`method_type`, `ag_key`, `compute`,
`config_default`, `can_hpo`, `is_bag`, task/problem-type/metric coverage), a suggested
`MethodMetadata` snippet, and — when an `info.py` metadata is passed — an **inferred-vs-provided
diff**. Use it to confirm `info.py` matches the raw data before processing. Two gotchas it catches:

- **`config_default` is compared post-rename**: configs are renamed to the method's prefix during
  processing, so a `config_default` authored with the raw prefix won't match. The snippet/diff shows
  the post-rename value — use that.
- **`method != suite`**: `process` fails if they're equal (suite defaults to method when unset).

Claude may run this itself if the venv (`tabarena[benchmark]`) and the data are available and it
helps resolve a field; otherwise hand it to the maintainer.

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

## Step 4: Build the command sheet

Assemble the exact commands with the resolved dotted reference. This is the block to hand the
maintainer (see the final report). Using `chimeraboost` as the example:

```bash
# 1. Inspect (confirm inferred fields match info.py)
python scripts/run_process_method.py <run_data_dir>

# 2. (Claude already edited info.py: suite + cache_type/cache_kwargs + date)

# 3. Process: build + cache metadata.yaml + processed/ + results/ locally
python scripts/run_process_method.py <run_data_dir> \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata --process

# 4. Upload dry-run: verifies every part exists locally and prints what/where (no creds needed)
python scripts/run_upload_results.py \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata

# 5. Real upload (R2 creds via env, NEVER flags):
python scripts/run_upload_results.py \
    --method-metadata tabarena.models.chimeraboost.info:chimeraboost_method_metadata --no-dry-run
```

Notes to pass along:
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

- **Edits Claude made**: the `info.py` upload fields (suite / cache_type / cache_kwargs / date /
  verified) and the `methods.py` import + collection entry.
- **The command sheet** from Step 4 (inspect → process → upload dry-run → upload), ready to run.
- **Open decisions / TODOs**: whether to flip `verified` to `True` (only after sign-off), and the
  caveat that the collection entry needs the real upload to complete before others can download it.
