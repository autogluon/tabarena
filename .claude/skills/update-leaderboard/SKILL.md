---
name: update-leaderboard
description: Regenerate the TabArena website artifacts and refresh a leaderboard Space's `data/` with them. Use this skill whenever a maintainer wants to publish the latest results to the leaderboard — e.g. "update the leaderboard", "regenerate the website artifacts and refresh the LB", "push the new results to the leaderboard Space", "refresh leaderboard-testing with the latest". Runs `scripts/run_generate_website_artifacts.py`, pre-checks that any newly added models classify correctly (Foundation Model / Tree-based / … not `❓ Other`), then swaps the generated artifacts into the Space repo's `data/` folder (deleting the old subtree first to dodge the stale-PNG gotcha) and bumps the version history in `website_texts.py`. Optionally serves the Space locally for preview. Complements `upload-method` (which publishes one method's results so they become downloadable — run that first).
argument-hint: <lb-code-dir> [<generation-venv>]
user-invocable: true
---

# Regenerate & Publish the TabArena Leaderboard

This skill drives the maintainer workflow that turns the **already-uploaded** benchmark results into
the tabarena.ai website artifacts and refreshes a leaderboard **Space repo's `data/`** with them.
It is the last stage of the lifecycle: `add-model` (integrate) → `benchmark-model` (run) →
`upload-method` (publish one method's results to r2, register it in `methods.py`) → **`update-leaderboard`**
(regenerate figures/tables + refresh the Space).

The authoritative prose lives in the module docstring of
`scripts/run_generate_website_artifacts.py` and the Space repo's `README.md`. This skill
operationalizes it and bakes in the two things that are easy to get wrong: the **model-type
pre-check** and the **stale-PNG gotcha**.

## What this skill delivers

1. A **model-type pre-check** so newly added models don't ship as `❓ Other`.
2. A **regeneration run** of `run_generate_website_artifacts.py` (Claude runs it — background +
   monitor; it's slow but needs no credentials).
3. A **refreshed `data/`** in the leaderboard Space repo, done the safe way (delete-then-copy), with
   sanity counts.
4. A **version-history bump** in the Space repo's `website_texts.py` (new dated entry + current-version line).
5. Optionally, a **local preview server** started with the Space repo's own `.venv`.
6. A **hand-off**: the maintainer commits + `git push`es the Space (Git LFS + Xet; no token on this box).

## Step 0: Gather inputs

Parse `$ARGUMENTS`. Collect (ask only for what's missing or ambiguous):

| Input | Example | Notes |
|---|---|---|
| `lb_code_dir` | `.../leaderboard-testing` | **Required.** The cloned HuggingFace **Space repo** (has `data/`, `main.py`, its own `.venv`). Use `leaderboard-testing` for a **safe private preview**; `leaderboard` is the **live production** Space. Default to `leaderboard-testing` and confirm before touching `leaderboard`. |
| `generation_venv` | `~/.venvs/tabarena_<date>` | A venv with **`tabarena[benchmark]`** installed (needed to run the generator — download + ray + figures). This is **not** the Space repo's `.venv` (which only has gradio/pandas). If unset, find the maintainer's under `~/.venvs/tabarena_*`. |
| new/changed models | `tabswift` | Which models were just added/uploaded — focus of the Step 1 pre-check. Infer from recent `git status`/`methods.py` diff if unstated. |
| arena | `tabarena` (main) | The main leaderboard uses `run_generate_website_artifacts.py` → `data/`. **BeyondArena** is a sibling (see the note at the end) → `run_generate_beyondarena_website_artifacts.py` → `data_beyondarena/`. This skill targets the main leaderboard unless told otherwise. |

## Step 1: Model-type pre-check (do this BEFORE generating)

The leaderboard's `Type` / `TypeName` columns are set at generation time by
`packages/tabarena/src/tabarena/website/website_format.py`. Any model whose family prefix isn't
registered there ships as `❓ Other`. Verify each new/changed model first — it's a 2-minute read that
saves a full regeneration.

How the classification works (read `website_format.py`):

- `add_metadata()` calls `get_model_family(config_type)`. `config_type` is the method's
  `model_key` (+ optional `name_suffix`); `model_key` **defaults to `ag_key`** (see
  `_method_metadata.py`).
- `get_model_family()` lowercases, **strips a leading `TA-`** (case-insensitive), then prefix-matches
  against `prefixes_mapping` (`foundational`, `neural_network`, `tree`, `baseline`, `reference`,
  `other`). No match → `Constants.other` (`❓ Other`).
- `get_rename_map()` gives the pretty display name (e.g. `TABSWIFT` → `TabSwift`).

Check, for each new model (example: TabSwift, `ag_key="TA-TABSWIFT"`):

1. Its `ag_key`/`model_key` prefix appears under the **intended** family list in
   `get_model_family`'s `prefixes_mapping` (TabSwift → `Constants.foundational`). The `TA-` strip
   means listing either `"TABSWIFT"` or `"TA-TABSWIFT"` works.
2. There's a `get_rename_map()` entry for a clean display name.

If missing, add the prefix to the right family list (and a rename entry). This is the same edit the
`add-model` skill calls out — cross-reference it. Confirm quickly:

```bash
<generation_venv>/bin/python -c "from tabarena.website.website_format import get_model_family; print(get_model_family('TA-TABSWIFT'))"
# -> Foundation Model   (NOT 'Other')
```

## Step 2: Regenerate the website artifacts (Claude runs; slow, no creds)

Run the generator **from `tabarena/scripts/`** (its `base_dir` is the relative
`generated_website_artifacts`, so cwd matters) with the **generation venv**. It's long-running —
`TabArenaContext.load_results(download_results="auto")` downloads the latest results, then figures +
tuning trajectories are built across CPUs with ray (a few minutes on many cores; longer here).
**Launch it in the background and poll the log** rather than blocking.

```bash
cd <tabarena>/scripts
nohup <generation_venv>/bin/python run_generate_website_artifacts.py > <scratch>/gen_website.log 2>&1 &
```

Monitor until the process exits, then verify outputs — **don't trust "exited" alone**:

- **Harmless noise to ignore:** at the end ray tears down its workers and each logs a
  `*** SIGTERM received ***` C++ stack trace (dozens of them, one per worker PID). A single Ray
  `FutureWarning` about accelerator env vars is also fine. Neither is a failure.
- **Real success signals** (check these):
  - `generated_website_artifacts/clean_website_artifacts/website_data/` exists with
    `imputation_no/` + `imputation_yes/`.
  - `clean_website_artifacts.zip` was written next to it.
  - In the generated CSVs, the new model appears with the right `TypeName` — re-confirm Step 1 held:
    ```bash
    cd <tabarena>/scripts/generated_website_artifacts/clean_website_artifacts/website_data
    <generation_venv>/bin/python - <<'PY'
    import pandas as pd
    df = pd.read_csv("imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv")
    print(df.loc[df["Model"].str.contains("TabSwift", case=False), ["Type","TypeName","Model"]].to_string(index=False))
    PY
    ```
  - Structural sanity in `website_data/`: **60** `website_leaderboard.csv`, **60** `n_datasets_*`
    markers, **240** `*.png.zip`, and **0** raw `*.png` (figures are zipped, not extracted).

## Step 3: Refresh the Space repo's `data/` (the stale-PNG gotcha)

`data/` in the Space repo mirrors the generated `website_data/`. **Delete the old subtree first, then
copy — do not overlay.** The app's `data_loading.unzip_png` returns an existing `.png` without
re-extracting its `.png.zip`, so a leftover unzipped `.png` (gitignored via `*.png`) makes the running
app serve the **stale** figure.

```bash
SRC=<tabarena>/scripts/generated_website_artifacts/clean_website_artifacts/website_data
DST=<lb_code_dir>/data

# 1. Sanity-check nothing but imputation_* lives in data/ (so the rm is safe):
find "$DST" -mindepth 1 -maxdepth 1 ! -name 'imputation_*'   # expect: no output

# 2. Delete the old subtree, then copy the fresh one in:
rm -rf "$DST"/imputation_*
cp -r "$SRC"/. "$DST"/

# 3. Verify the swap:
echo "raw .png (MUST be 0): $(find "$DST" -name '*.png' | wc -l)"
echo ".png.zip:             $(find "$DST" -name '*.png.zip' | wc -l)"   # 240
echo "csv:                  $(find "$DST" -name 'website_leaderboard.csv' | wc -l)"  # 60
```

Then confirm the diff is clean — **all modifications, no adds/deletes/untracked** (a new or removed
subset would show up here and means the layout changed):

```bash
cd <lb_code_dir> && git status --short | awk '{print $1}' | sort | uniq -c   # expect only 'M'
```

(The Space's `README.md` describes an equivalent "unzip `clean_website_artifacts.zip` into `data`"
path — the delete-then-copy above is the same result done safely.)

## Step 4: Bump the version history (Claude does now)

Every leaderboard refresh gets a new entry in the Space repo's **`website_texts.py`** —
the `VERSION_HISTORY_BUTTON_TEXT` block that the UI's "Version History" button renders. **Don't skip
this**: it's the user-facing record of what changed, and it's easy to forget because it lives in the
Space repo, not in `tabarena`. Read the block first, then `Edit`:

1. **Add a dated entry at the top of the list** (newest first; date format `YYYY/MM/DD`, today's date)
   with a bumped version number, describing what changed. Match the existing wording:
   - New model → `Add new verified model: <Name>` or `Add new unverified model: <Name>`. Pick
     **verified vs unverified from the model's `info.py` `verified` flag** (`verified=False` →
     "unverified"). List multiple on one line if several shipped together.
   - Other changes (UI, metric, reference pipeline, removals) → mirror the phrasing of past entries.
2. **Bump `**Current Version: TabArena-vX.Y.Z**`** at the top of the block to the same new number.

Version bumping: increment the **last** component for a normal model-addition / data refresh
(`v0.1.5.2` → `v0.1.5.3`); larger jumps (`v0.1.5` → new UI, `v0.1.6`) are for bigger releases — match
the granularity of comparable past entries.

Example (adding the unverified TabSwift on 2026/07/10, bumping `v0.1.5.2` → `v0.1.5.3`):

```
**Current Version: TabArena-v0.1.5.3**
...
* 2026/07/10-v0.1.5.3:
    * Add new unverified model: TabSwift
* 2026/07/08-v0.1.5.2:
    * Add new verified model: TabFM
```

## Step 5: Preview locally (optional)

Start the app with the **Space repo's own `.venv`** (it has gradio + `gradio_leaderboard`; the
generation venv does not). `main.py`'s `launch()` binds `127.0.0.1:7860`.

```bash
cd <lb_code_dir>
nohup .venv/bin/python main.py > <scratch>/lb_serve.log 2>&1 &
```

Gradio block-buffers stdout to a file, so the log may stay empty — confirm it's up by the port, not
the log:

```bash
ss -tlnp | grep 7860 ; curl -s -o /dev/null -w "HTTP %{http_code}\n" http://127.0.0.1:7860/
```

On a remote box the maintainer needs port-forwarding to view it (VS Code forwards `7860`
automatically, or `ssh -L 7860:localhost:7860 …`).

## Step 6: Hand off (maintainer commits + pushes)

Claude does **not** commit/push the Space — it's a HuggingFace Space using **Git LFS + Git Xet** for
the `.png.zip` files, and there is no HF token on this box. Tell the maintainer to, from
`<lb_code_dir>`:

```bash
git add data website_texts.py && git commit -m "Update leaderboard data + version history" && git push
```

Surface these caveats:
- The `data/` swap only reflects methods whose results were actually **uploaded** (`upload-method`,
  the real `--no-dry-run`). A method registered in `methods.py` but not uploaded won't have artifacts.
- `leaderboard-testing` is the **private preview**; pushing to `leaderboard` publishes **live**.
- If the push is rejected for storage, see the generator's module docstring (Space "Git Storage
  Usage" cleanup — destructive).

## Note: BeyondArena is a sibling flow

The second leaderboard has its own generator and target folder:
`scripts/run_generate_beyondarena_website_artifacts.py` →
`generated_beyondarena_website_artifacts/clean_website_artifacts/` (`subsets/` + `result_plots/`) →
Space repo's **`data_beyondarena/`**. Same delete-then-copy discipline applies. Only touch it when the
maintainer asks for BeyondArena; this skill's default is the main leaderboard `data/`.
