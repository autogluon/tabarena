---
name: update-leaderboard
description: Regenerate the TabArena website artifacts and refresh a leaderboard Space's `data/` with them. Use this skill whenever a maintainer wants to publish the latest results to the leaderboard — e.g. "update the leaderboard", "regenerate the website artifacts and refresh the LB", "push the new results to the leaderboard Space", "refresh leaderboard-testing with the latest". Runs `scripts/run_generate_website_artifacts.py`, pre-checks that any newly added *models* classify correctly (Foundation Model / Tree-based / … not `❓ Other`; systems are typed from `method_class` and need no pre-check), then swaps the generated artifacts into the Space repo's `data/` folder (deleting the old subtree first) and bumps the version history in `website_texts.py`. Optionally serves the Space locally for preview. Complements `upload-method` (which publishes one method's results so they become downloadable — run that first).
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
operationalizes it and bakes in the thing that is easy to get wrong: the **model-type pre-check**.

## What this skill delivers

1. A **model-type pre-check** so newly added models don't ship as `❓ Other`. (Systems are exempt: they
   are typed from `MethodMetadata.method_class`, not from a name prefix.) Next to it, a logo pre-check
   so a new organization does not ship as a letter tile.
2. A **regeneration run** of `run_generate_website_artifacts.py` (Claude runs it — background +
   monitor; about two minutes, no credentials).
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
  against `prefixes_mapping` (`foundational`, `neural_network`, `tree`, `baseline`, `other`). No match
  → `Constants.other` (`❓ Other`).
- **Systems skip all of this.** When `method_class == "system"`, `add_metadata` sets the family to
  `Constants.system` (📊) directly, and the figure paths recognize them from `system_display_names`.
  There is no prefix to register, so this pre-check only concerns models.
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

### Step 1b: Organization logos

The table shows each method's `organization` as a logo. The logos live in the Space repo, not here:
the files in `<lb_code_dir>/assets/logos/` and the name-to-file map `ORGANIZATION_LOGOS` in
`<lb_code_dir>/views.py`. An organization missing from the map shows its initial instead. List the
registered organizations without a logo (reads `views.py` with `ast`, so gradio is not needed):

```bash
cd <tabarena> && <generation_venv>/bin/python -P - <lb_code_dir>/views.py <<'PY'
import ast, pathlib, sys
from tabarena.models import get_model_registry
from tabarena.systems._registry import get_system_registry
infos = [*get_model_registry().values(), *get_system_registry().values()]
orgs = {i.method_metadata.organization for i in infos} - {None}
tree = ast.parse(pathlib.Path(sys.argv[1]).read_text())
logos = next(ast.literal_eval(n.value) for n in tree.body
             if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "ORGANIZATION_LOGOS")
print("organizations without a logo:", sorted(orgs - logos.keys()) or "none")
PY
```

For each one, take the logo attached to the method's PR (the PR template asks for it), save it under
`assets/logos/` with a lowercase file name, and add the map entry, spelled exactly as in `info.py`.
Check it in both themes in the preview (Step 5); a logo that vanishes on the dark table gets a white
tile baked into the file, like the few existing ones that need it. When the PR has no logo, ship with the
initial and draft a comment asking the submitter for one; post it only after the maintainer says so.

## Step 2: Regenerate the website artifacts (Claude runs; about two minutes, no creds)

Run the generator **from `tabarena/scripts/`** (its output folder is the relative
`generated_website_artifacts`, so cwd matters) with the **generation venv**.
`TabArenaContext.load_results(download_results="auto")` downloads the latest results, then
`tabarena.website.build.build_website` computes every cell of the grid in one ray pass (about two
minutes on 96 cores). **Launch it in the background and poll the log** rather than blocking.

```bash
cd <tabarena>/scripts
nohup <generation_venv>/bin/python run_generate_website_artifacts.py > <scratch>/gen_website.log 2>&1 &
```

Flags: `--elo-bootstrap-rounds 1` for a toy run (Elo CIs meaningless); `--full-figures` also renders
every cell's paper figure suite (static figures, LaTeX, GIFs) into `raw_website_artifacts/`, which is
paper material, is not published, and takes several times as long.

Monitor until the process exits, then verify outputs — **don't trust "exited" alone**:

- **Harmless noise to ignore:** at the end ray tears down its workers and each logs a
  `*** SIGTERM received ***` C++ stack trace (dozens of them, one per worker PID). A single Ray
  `FutureWarning` about accelerator env vars is also fine. Neither is a failure.
- **Real success signals** (check these):
  - The log ends with `Website data written to generated_website_artifacts/website_data in <N>s`.
  - `generated_website_artifacts/website_data/` holds exactly what the Space's TabArena `data/` holds:
    the `entrants_*` roots with one `website_leaderboard.csv` per cell, `n_datasets.json`, and
    `pages.sqlite` (every cell's table and explorer pages in one file, about 15 MB; see
    `tabarena.website.page_pack`). The build empties the folder first, so nothing from an earlier
    run survives.
  - In the generated CSVs, the new model appears with the right `TypeName` — re-confirm Step 1 held:
    ```bash
    cd <tabarena>/scripts/generated_website_artifacts/website_data
    <generation_venv>/bin/python - <<'PY'
    import pandas as pd
    df = pd.read_csv("entrants_models/imputation_no/splits_all/tasks_all/datasets_all/website_leaderboard.csv")
    print(df.loc[df["Model"].str.contains("TabSwift", case=False), ["Type","TypeName","Model"]].to_string(index=False))
    PY
    ```
  - Structural sanity: **16** `entrants_*` roots (one per combination of the system categories, and
    each one's `_noleak` twin computed without the methods that carry a leakage warning), and as
    many CSVs as `n_datasets.json` has keys and as the Space has now (1,856 since v0.1.9.12), unless
    a subset was added or removed: `diff <(cd $SRC && find entrants_* -type f | sort) <(cd $DST &&
    find entrants_* -type f | sort)` is empty then. The pack lists six pages per cell plus the
    per-dataset browser of each pool's unrestricted cells:
    `python -P -c "from tabarena.website.page_pack import read_page_pack; v, p = read_page_pack('pages.sqlite'); print(v, len(p))"`
    (9,344 today).
  - Pool sanity: `entrants_models` must contain **no** AutoGluon row, and a model's Elo must differ
    between `entrants_models` and `entrants_open_llm_api` (a wider field re-rates everyone). If they
    match, the pool filter did not apply.
    A method with `leakage_evidence` (LimiX-2) must appear in `entrants_models` with a non-empty
    `Leakage` cell and be absent from `entrants_models_noleak`, where the others' Elo differs.

## Step 3: Refresh the Space repo's `data/`

`data/` in the Space repo mirrors the generated `website_data/`. **Delete the old TabArena files
first, then copy — do not overlay**, so a subset that no longer exists cannot survive as a leftover.

```bash
SRC=<tabarena>/scripts/generated_website_artifacts/website_data
DST=<lb_code_dir>/data

# 1. Sanity-check nothing else lives in data/ (so the rm is safe). A checkout from before the page
#    pack still has explorer HTML files under entrants_*; the rm takes them along.
find "$DST" -mindepth 1 -maxdepth 1 ! -name 'entrants_*' ! -name 'imputation_*' ! -name 'n_datasets.json' ! -name 'pages.sqlite'   # expect: no output

# 2. Delete the old files, then copy the fresh ones in.
rm -rf "$DST"/entrants_* "$DST"/imputation_* "$DST"/n_datasets.json "$DST"/pages.sqlite
cp -r "$SRC"/. "$DST"/

# 3. Verify the swap:
echo "entrant pools: $(find "$DST" -mindepth 1 -maxdepth 1 -name 'entrants_*' | wc -l)"  # 16
echo "csv:           $(find "$DST" -name 'website_leaderboard.csv' | wc -l)"  # same as before the swap unless a subset changed
echo "html (MUST be 0): $(find "$DST" -name '*.html' | wc -l)"  # the pages live in pages.sqlite
```

Then confirm the diff: modified CSVs, `n_datasets.json` and `pages.sqlite` only, nothing added or
deleted (a new or removed subset would show up here and means the layout changed). The first update
after the switch to the pack is the exception: it deletes every `*.html` and adds `pages.sqlite`.

```bash
cd <lb_code_dir> && git status --short | awk '{print $1}' | sort | uniq -c   # expect only 'M'
```

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
   - Other changes (UI, metric, system, removals) → mirror the phrasing of past entries.
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

By default the **maintainer** commits/pushes the Space (Git LFS + Git Xet for `data/pages.sqlite` and
BeyondArena's `.png.zip` files;
`hf auth login --add-to-git-credential` is set up on this box, so Claude *can* push when explicitly
asked). From `<lb_code_dir>`:

```bash
git add data website_texts.py && git commit -m "Update leaderboard data + version history" && git push
```

`git add data` matters: an IDE commit of "changed files only" misses files a refresh adds (a new
subset's CSVs) or deletes.

Surface these caveats:
- The `data/` swap only reflects methods whose results were actually **uploaded** (`upload-method`,
  the real `--no-dry-run`). A method registered in `methods.py` but not uploaded won't have artifacts.
- `leaderboard-testing` is the **private preview**; pushing to `leaderboard` publishes **live**.

### Storage-limit rejections (`Repository storage limit reached (Max: 1 GB)`)

Space repos cap git-LFS storage at 1 GB and each refresh adds ~85 MB of new figure zips, so this
rejection recurs every ~10 refreshes. Fix it with the **HEAD-aware purge** script next to this skill
— it deletes only the stored LFS objects the local HEAD (the state about to be pushed) no longer
references, so the live revision (incl. `data_beyondarena/`) keeps working, and it leaves history
un-rewritten so the pending push stays a fast-forward. Old Space revisions permanently lose their
binaries (fine — artifacts are regenerable):

```bash
cd <lb_code_dir>   # HEAD must be the commit you are about to push
<generation_venv>/bin/python <tabarena>/.claude/skills/update-leaderboard/purge_stale_lfs.py            # dry run
<generation_venv>/bin/python <tabarena>/.claude/skills/update-leaderboard/purge_stale_lfs.py --delete   # purge, then push
```

## Note: BeyondArena is a sibling flow

The second leaderboard has its own generator and target folder:
`scripts/run_generate_beyondarena_website_artifacts.py` →
`generated_beyondarena_website_artifacts/clean_website_artifacts/` (`subsets/` + `result_plots/`, and
their twins without the methods with a leakage warning, `subsets_noleak/` + `result_plots_noleak/`) →
Space repo's **`data_beyondarena/`**. Same delete-then-copy discipline applies. Only touch it when the
maintainer asks for BeyondArena; this skill's default is the main leaderboard `data/`.

The generator crosses four filter axes (split, size, features, tasks) and publishes every cell that
holds at least one dataset, under `subsets/split_*/size_*/features_*/tasks_*/` (354 of the 500 cells
today). The cells run in parallel with ray (about a minute and a half on 96 cores);
`--skip-generate` re-runs only the convert step. Copy `n_datasets.json` (each cell's dataset count)
along with the folders. Delete `subsets*/` and `result_plots*/` before
copying: the unzipped `*.png` files that `data_loading.unzip_png` leaves there are gitignored and
would be served stale. Then compare the file list against the Space's tracked one
(`diff <(cd $SRC && find . -type f | sed 's|^\./||' | sort) <(cd $DST && git ls-files | sort)` is empty when no cell
changed). Each cell must carry a `leaderboard_table.html` (one per `website_leaderboard.csv`), which
the app embeds as the full table, and only the all-`all` cell carries `per_dataset_explorer.html`.
