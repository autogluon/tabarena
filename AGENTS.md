# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs live in `README.md`.

## Project Overview

TabArena is a living benchmark for tabular machine learning. It evaluates ML methods across 51 curated datasets with cross-validated ensembles, HPO simulation, and leaderboard generation. Built on top of AutoGluon.

## Repository Layout

This repo is a **uv workspace** (root `pyproject.toml`). Its installable packages live under `packages/` (workspace members `tabarena`, `bencheval`, `tabflow_slurm`), alongside supporting dirs:

- `packages/tabarena/` — Core package. Repository pattern for benchmark data, model wrappers (`src/tabarena/models/`), system wrappers (`src/tabarena/systems/`), simulation, evaluation, plotting. Depends on AutoGluon and `bencheval`.
- `packages/bencheval/` — Standalone lightweight metrics/leaderboard package (ELO, win-rates, ranks, improvability). Computes leaderboards from results DataFrames. No dependency on `tabarena`.
- `packages/tabflow_slurm/` — Package (own `pyproject.toml`, a uv-workspace member) for running experiments on SLURM clusters. Depends on `tabarena`. See `packages/tabflow_slurm/README.md` and `packages/tabflow_slurm/AGENTS.md`.
- `examples/` — Usage examples for benchmarking, plotting, meta-learning, custom models.
- `tests/` — All tests, grouped by package (`tests/tabarena/`, `tests/bencheval/`, `tests/tabflow_slurm/`, mirroring each package's `src/` layout) plus `tests/integration/` for cross-package tests.

## Setup Commands

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). This is a uv *virtual workspace*
(the root `pyproject.toml` has no `[project]` table), so install the `tabarena` package directly
from `packages/tabarena` rather than running `uv sync` at the root. `--prerelease=allow` is required
for the pre-release AutoGluon dependency. From the repo root, after creating/activating a venv
(`uv venv --seed --python 3.12 && source .venv/bin/activate`):

```bash
uv pip install --prerelease=allow -e "./packages/tabarena"               # Minimal: evaluation/leaderboard/metrics only
uv pip install --prerelease=allow -e "./packages/tabarena[plot]"         # + leaderboard/result plotting
uv pip install --prerelease=allow -e "./packages/tabarena[text]"         # + semantic text features (sentence-transformers; pulls torch)
uv pip install --prerelease=allow -e "./packages/tabarena[preprocessing]" # + skrub datetime/statistical-text feature generators
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"    # Full install (models + plot + text + preprocessing)
```

The core install is intentionally minimal (issue #323): it depends on `autogluon.tabular`
(not the full `autogluon` meta-package) and leaves plotting, text embeddings, and skrub
feature generators to the extras above (all imported lazily). `[benchmark]` is the union.

For editable AutoGluon development (one directory up):

```bash
../autogluon/full_install.sh
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"
```

## Lint & Format

```bash
ruff check .               # Lint (config: ruff.toml)
ruff format .              # Format
```

Key rules: `from __future__ import annotations` is required in every file (enforced via isort `required-imports`). Line length 120 (the formatter is the authority — `E501` is not enforced). Google-style docstrings. Run ruff on touched files before finishing a task.

**CI runs `ruff check .` and `ruff format --check .`** (see `.github/workflows/pytest-pytest.yml`), so lint/format violations fail the build. Optionally install the local pre-commit hook so commits are auto-fixed before they reach CI:

```bash
pip install pre-commit && pre-commit install   # one-time, per clone
```

After that, `git commit` runs ruff on staged files; if a hook reformats or fixes anything the commit aborts — re-`git add` and commit again. The ruff version is pinned identically in `.pre-commit-config.yaml`, the CI workflow, and the `lint` dependency group in `packages/tabarena/pyproject.toml`; keep all three in sync.

## Testing

Tests live in a single top-level `tests/` dir, organized to mirror `src/`
(tabarena areas as subfolders — `metrics/`, `repository/`, `benchmark/`, `models/`,
… — plus `tests/bencheval/` and `tests/tabflow_slurm/` for the other two packages).
The root `pyproject.toml` `[tool.pytest.ini_options]` sets `testpaths = ["tests"]`,
so a bare `pytest` from the repo root runs the whole suite.

```bash
pytest                                      # All tests
pytest tests/metrics/test_metrics.py        # Single file
pytest -k test_name -x                      # Single test, stop on failure
pytest tests/bencheval                      # One package's tests
```

The default `pytest` deselects two slow/fragile groups via `addopts`
(`-m 'not network and not models'`):

- **`network`** — tests that hit the network (e.g. download a Hugging Face model).
- **`models`** — `test_all_models.py`, which fits every registered model via
  AutoGluon's `FitHelper`. It is parametrized over the model registry and skips
  models whose optional deps aren't installed (`ImportError`) or that need a GPU
  (`compute='gpu'`, no CUDA). Run one model with `pytest -m models -k TabM`, or
  the whole sweep with `pytest -m models` (needs `tabarena[benchmark]`).

Both groups run in the nightly workflow. CI's per-PR job (`.github/workflows/pytest-pytest.yml`)
runs `pytest` on Python 3.11 against `./packages/tabarena[plot,preprocessing,data-foundry]`
plus the editable `tabflow_slurm` package (so its tests and the data_foundry-gated tests run),
but **not** `[text]`/`[benchmark]` — so it stays fast (no model fitting, no torch).

## Architecture

### Core data flow

```
Raw predictions → EvaluationRepository → Simulation/Portfolio → Results DataFrames → TabArena leaderboard (bencheval)
```

### Key abstractions

- **`EvaluationRepository`** (`packages/tabarena/src/tabarena/repository/evaluation_repository.py`) — Central class combining config metadata/rankings (`ZeroshotSimulatorContext`), cached val/test predictions (`TabularModelPredictions`), and `GroundTruth`. Supports subsetting by datasets/folds/configs/problem_types and ensemble selection via mixins.
- **`TabularModelPredictions`** (`packages/tabarena/src/tabarena/predictions/`) — Abstract base for prediction storage. Implementations: `TabularPredictionsInMemory` (dict-based) and `TabularPredictionsMemmap` (disk-based memory-mapped for large benchmarks). Structure: `{dataset: {fold: {val/test: {config: predictions}}}}`.
- **`AbstractExecModel`** (`packages/tabarena/src/tabarena/benchmark/exec_models/base.py`) — Base for the benchmark *execution* wrappers (the AutoGluon wrappers live in `benchmark/exec_models/autogluon.py`). New benchmarked models live in one folder per model at `packages/tabarena/src/tabarena/models/<model>/` (`model.py` = AutoGluon wrapper subclassing AG's `AbstractModel`/`AbstractTorchModel`, `hpo.py` = search-space generator, `info.py` = `ModelInfo`/`MethodMetadata` registry entry), auto-discovered by `packages/tabarena/src/tabarena/models/_registry.py::discover_models()` (which `packages/tabarena/src/tabarena/benchmark/exec_models/registry.py` then derives the AG registry from). Use the **`add-model` skill** — there is no `benchmark/models/ag/<model>/` layout for new models.
- **`ExternalSystemModel`** (`packages/tabarena/src/tabarena/benchmark/exec_models/external.py`) — Base for a benchmarked **system**: a whole pipeline that picks, tunes and ensembles models inside its own budget (AutoGluon, TabFM+, an agent, a hosted API). Systems live one folder per system at `packages/tabarena/src/tabarena/systems/<system>/` (`system.py` = the `ExternalSystemModel` subclass, `hpo.py` = the `SystemConfigGenerator`, `info.py` = `SystemInfo`/`MethodMetadata`), auto-discovered by `systems/_registry.py::discover_systems()`. See "Systems" below.
- **`ExperimentRunner` / `ExperimentBatchRunner`** (`packages/tabarena/src/tabarena/benchmark/experiment/`) — Execute model fitting across tasks. Configured via YAML (`experiment_constructor.py`).
- **`ZeroshotSimulatorContext`** (`packages/tabarena/src/tabarena/simulation/`) — Manages config rankings for HPO simulation and portfolio generation.
- **`BenchmarkEvaluator`** (`packages/bencheval/src/bencheval/evaluator.py`) — Leaderboard computation from results DataFrames. Independent of the core `tabarena` package.
- **`EntrantPool`** (`packages/tabarena/src/tabarena/evaluation/entrants.py`) — One field of competitors, evaluated together. The leaderboard publishes one artifact tree per pool. See "Entrant pools" below.

### Systems

A **model** is one method run under TabArena's shared tuning protocol. A **system** manages its own budget, model selection and ensembling; benchmarking it means handing it the data and the constraints and recording what comes back.

Two fields on `MethodMetadata` carry this, both orthogonal to `method_type` (which stays a *result-shape* discriminator: which parquet `load_results` reads, whether HPO simulation applies):

- `method_class` — `"model"` (default) or `"system"`. Not inferable from raw results, which record a system exactly like any other `method_type="baseline"` run, so a system must declare it. Build the metadata with `MethodMetadata.system(...)`.
- `tags` — from the `MethodTag` vocabulary, for what a reader must weigh before trusting a comparison. Keep it small; a tag earns its place only when someone would reasonably exclude a method because of it.
  - `with-llm` — an LLM is involved somewhere, agents included. Says nothing about whether the system is open-source.
  - `closed-source-api` — the method runs behind a remote API we cannot inspect and whose behavior can change. Being an API implies closed-source here, so this is one tag rather than two.

Adding a system mirrors `add-model`: create `systems/<key>/` with `system.py`, `hpo.py` and `info.py`, and export a `<key>_info: SystemInfo` that `discover_systems()` picks up. Systems stay out of the AutoGluon model registry on purpose (no `ag_key`, no search space) and run through the experiment bundle's `system_experiments=True` mode. `SystemInfo.__post_init__` rejects metadata that is not `method_class="system"`, so a misdeclared system fails at import rather than misclassifying downstream.

The website types a system from `method_class`, never from its name. `website_format.get_model_family` classifies models by name prefix, but a system short-circuits to the `System` family via the declared set (`system_display_names`), so a new system never lands in `❓ Other`.

### Entrant pools

Every headline number is relative to who competed: Elo is pairwise over the participants, `improvability` is `1 - best_error_in_pool / error`, and the ranks are positions in the field. Narrowing the field therefore needs a recompute, not a row filter, which is why the entrant pool is a subset axis alongside imputation and splits rather than a toggle on the published table.

`evaluation/entrants.py` groups systems into independently selectable `SYSTEM_CATEGORIES`: `open` (untagged, i.e. open-source and local), `llm` (`with-llm`), and `api` (`closed-source-api`). Every combination is published as its own pool, so there are `2 ** len(SYSTEM_CATEGORIES)` = 8, keyed `models`, `open`, `llm`, `api`, `open_llm`, `open_api`, `llm_api`, `open_llm_api`. Models always compete.

Independent rather than a cumulative ladder because "LLM-based systems but not the plain open-source ones" is a question people actually have, and a ladder cannot express it. A system carrying several tags belongs to several categories and needs *all* of them selected, so a closed-API LLM system never appears on the strength of a property the reader excluded.

`evaluation/subset_grid.py` crosses the pools with the other axes into a 480-cell grid and owns the folder layout (`entrants_<key>/imputation_.../splits_.../tasks_.../datasets_...`), which the leaderboard Space's `Subset.rel_path` mirrors segment for segment. Adding a fourth category doubles the artifact count and the generation time; that is the price of independent toggles.

One coupling to know about when you register a system: `eval_all.get_pool_reference_lines` returns the pool's admitted systems as `(baselines, baseline_colors)`, and `baselines` is not only what the figures draw as horizontal reference lines. `LeaderboardReporter.eval` keeps a row only when its method maps to a config framework type *or* is named in `baselines`, and a system is neither, so a system missing from that list is deleted from the pool's published numbers with nothing logged. It is derived from `method_metadata_info` for exactly that reason, and `tests/tabarena/evaluation/test_entrants.py` guards that the widest pool covers every system in the shipped collection.

### Data caching

TabArena uses five independent caches. Configure them all at once with `tabarena.caching.CacheConfig` — the single, documented surface (`TabArenaContext(cache_config=CacheConfig.from_root(...))`; the context applies it on construction and re-applies it inside `run_jobs`, so distributed workers inherit it). The SLURM path uses the **same** object: the setup embeds `context.cache_config` in the `JobBatch`, and each worker applies it (no `--openml_cache_dir` wiring). See the `CacheConfig` docstring for the authoritative per-cache reference.

| Cache | `CacheConfig` field | Holds | Set via | Default |
|---|---|---|---|---|
| OpenML (most important) | `openml` | Materialized datasets + CV splits + all TabArena-derived task artifacts (`tabarena_tasks/`, `tabarena_text_cache/`, `tabarena_metadata_cache/`, `local/datasets/`) | `openml.config.set_root_cache_directory` (no env var) | `~/.cache/openml` |
| HuggingFace | `huggingface` | Foundation-model weights (TabPFN / Mitra / LimiX / ... + text-embedding models) | `HF_HOME` | `~/.cache/huggingface/hub` |
| Data Foundry | `data_foundry` | The one-time raw dataset download (data_foundry/BeyondArena), later materialized into the OpenML cache. **Not** `HF_HOME` — data_foundry passes an explicit `cache_dir` to `snapshot_download` | `DATA_FOUNDRY_CACHE` | `~/.cache/data_foundry` |
| TabArena | `tabarena` | Results / baselines / leaderboard artifacts (~100 GB raw, ~10 GB processed, <1 MB results per method) | `set_tabarena_cache_root` / `TABARENA_CACHE` | `~/.cache/tabarena` |
| Results (run output) | `results` | The runner's `expname` (`{expname}/data/{method}/{task}/{repeat}_{fold}/results.pkl`) | `run_jobs(expname=...)` | throwaway temp dir |

### Processing & uploading method artifacts (maintainers)

Turn a benchmark run's already-present raw `results.pkl` files into cached, hosted, and registered TabArena artifacts — no download, no auto-generation. Steps 1–2 are single-method CLIs (full flag reference in each script's module docstring); step 3 is a small code change.

1. **Author + process** — `scripts/run_process_method.py` (logic in `tabarena.tools.process_local_raw_data`):
   - Inspect the raw dir for a suggested metadata snippet: `python scripts/run_process_method.py <run_dir>/data` (recursive; prints the inferred fields + a `MethodMetadata.config/.baseline/.portfolio(...)` snippet). Paste it into `packages/tabarena/src/tabarena/models/<model>/info.py` and fill in `suite` (required, must differ from `method`) + the manual fields.
   - Process: append `--method-metadata tabarena.models.<model>.info:<x>_method_metadata --process`. **Processing requires an explicit `MethodMetadata`**, verified against the raw data first (method_type / compute / ag_key / `config_default` / can_hpo / is_bag must align, and `method != suite`); a `method`-name mismatch only warns, other mismatches error unless `--ignore-metadata-mismatch`. Caches `metadata.yaml` + `processed/` + `results/` (plus raw + HPO trajectories, on by default) under the TabArena cache.
2. **Upload to r2** — `scripts/run_upload_results.py`:
   - Dry-run (default) verifies each part exists locally and prints what/where: `python scripts/run_upload_results.py --method-metadata tabarena.models.<model>.info:<x>_method_metadata` (or `--from-cache METHOD SUITE` to load from the local cache).
   - Real upload: add `--no-dry-run` with `R2_ACCOUNT_ID` / `R2_ACCESS_KEY_ID` / `R2_SECRET_ACCESS_KEY` set in the environment (never as flags). **r2 only** — the metadata needs `cache_type="r2"` + `cache_kwargs={"bucket", "prefix"}`; the dry-run prints the exact `--no-dry-run` command and, when the creds are unset, how to obtain them (`MethodMetadata.r2_credentials_help()`). `raw` uploads by default (`--no-upload-raw` to skip).
3. **Register in the appropriate context's collection** — add the method so it appears in the benchmark. For TabArena, import the model's `info.py` `method_metadata` and add it to `tabarena_method_metadata_collection` in `packages/tabarena/src/tabarena/contexts/tabarena/methods.py` (the collection lists each model's `info.py` metadata directly, and is itself the paper method set used by `TabArenaContext`). It flows into `tabarena_method_metadata_complete_collection` automatically. Other arena contexts register in their own collection.


### Example
```bash

# Verify local cache state
python scripts/run_process_method.py /path/to/nori_regression_18062026/data

# Edit metadata in info.py

 python scripts/run_process_method.py /path/to/nori_regression_18062026/data --method-metadata tabarena.models.nori.info:nori_method_metadata --process

python scripts/run_upload_results.py --method-metadata tabarena.models.nori.info:nori_method_metadata

python scripts/run_upload_results.py --method-metadata tabarena.models.nori.info:nori_method_metadata --no-dry-run

# Add model to the collection in `packages/tabarena/src/tabarena/contexts/tabarena/methods.py`
```

## Conventions

- **Add a new model**: create one folder `packages/tabarena/src/tabarena/models/<model>/` (`model.py`, `hpo.py`, `info.py`, `__init__.py`), then edit `models/__init__.py` (lazy class entry), `models/utils.py` (name→generator map), and `packages/tabarena/pyproject.toml` (a per-model extra). The registry auto-discovers the model from its `info.py`, and `tests/tabarena/models/test_all_models.py` then fits it automatically — there is **no per-model test file**. Only add an entry to `tests/tabarena/models/smoke_configs.py` if the smoke fit needs faster toy hyperparameters or a restricted problem-type set (keyed by the model's `MethodMetadata.method`). **Use the `add-model` skill**, which encodes this and points to reference implementations (foundation / torch / sklearn).
- **Imports**: `from __future__ import annotations` must be the first import in every `.py` file. Use absolute imports rooted at the package (e.g., `from tabarena.repository import EvaluationRepository`).
- **Optional dependencies**: each model has its own pyproject extra under `packages/tabarena/pyproject.toml`; the `benchmark` extra is the union. Heavy/optional libs must never be imported at module top-level in core paths — import inside the model wrapper.
- **Scoping jobs**: when scoping `context.build_jobs` / `build_and_run_jobs` (in examples and code), pass a typed `TaskSubset` via `task_subset=` — it is the single source of truth for the available filters (`subset`, `dataset_names`, `split_indices`, `problem_types`, `n_train_samples`, ...). Prefer it over the loose `dataset_names=` / `build_kwargs={...}` keyword conveniences.
- **No new top-level docs files** unless the user asks. Edit existing files in place.
- **Prose style**: everything you write for a human reader (docstrings, comments, markdown, commit messages, PR descriptions, chat replies) follows [AI Writing Tropes to Avoid](#ai-writing-tropes-to-avoid) at the bottom of this file.

## PR & Commit Guidance

- Keep commits focused; do not bundle unrelated refactors with bug fixes.
- Run `ruff check .` and `pytest` on affected paths before opening a PR.
- CI is mandatory on `main` PRs.

## Things to Avoid

- Do not add a `tst/` dir or per-package `tests/` dirs — all tests live in the single top-level `tests/`, grouped by package (`tests/tabarena/`, `tests/bencheval/`, `tests/tabflow_slurm/`, `tests/integration/`).
- Do not import optional model dependencies at the top of shared modules; lazy-import inside the wrapper.
- Do not skip `from __future__ import annotations` — ruff will fail CI.
- Do not change the public API of `EvaluationRepository`, `TabularModelPredictions`, or `bencheval.evaluator.BenchmarkEvaluator` without explicit user direction; they are consumed by external scripts and artifacts.

---

# AI Writing Tropes to Avoid

Applies to everything you write in this repo that a human reads: docstrings,
comments, markdown docs, commit messages, PR descriptions, user-facing copy,
and your replies in the chat.

Source: [tropes.fyi](https://tropes.fyi) by [ossama.is](https://ossama.is)

---

## Word Choice

### "Quietly" and Other Magic Adverbs

Overuse of "quietly" and similar adverbs to convey subtle importance or understated power. AI reaches for these adverbs to make mundane descriptions feel significant. Also includes: "deeply", "fundamentally", "remarkably", "arguably".

**Avoid patterns like:**
- "quietly orchestrating workflows, decisions, and interactions"
- "the one that quietly suffocates everything else"
- "a quiet intelligence behind it"

### "Delve" and Friends

Used to be the most infamous AI tell. "Delve" went from an uncommon English word to appearing in a staggering percentage of AI-generated text. Part of a family of overused AI vocabulary including "certainly", "utilize", "leverage" (as a verb), "robust", "streamline", and "harness".

**Avoid patterns like:**
- "Let's delve into the details..."
- "Delving deeper into this topic..."
- "We certainly need to leverage these robust frameworks..."

### "Tapestry" and "Landscape"

Overuse of ornate or grandiose nouns where simpler words would do. "Tapestry" is used to describe anything interconnected. "Landscape" is used to describe any field or domain. Other offenders: "paradigm", "synergy", "ecosystem", "framework".

**Avoid patterns like:**
- "The rich tapestry of human experience..."
- "Navigating the complex landscape of modern AI..."
- "The ever-evolving landscape of technology..."

### The "Serves As" Dodge

Replacing simple "is" or "are" with pompous alternatives like "serves as", "stands as", "marks", or "represents". AI avoids basic copulas because its repetition penalty pushes it toward fancier constructions (I've studied this!).

**Avoid patterns like:**
- "The building serves as a reminder of the city's heritage."
- "Gallery 825 serves as LAAA's exhibition space for contemporary art."
- "The station marks a pivotal moment in the evolution of regional transit."

---

## Sentence Structure

### Negative Parallelism

The "It's not X -- it's Y" pattern, often with an em dash. The single most commonly identified AI writing tell. Man I f*cking hate it. AI uses this to create false profundity by framing everything as a surprising reframe. One in a piece can be effective; ten in a blog post is a genuine insult to the reader. Before LLMs, people simply did not write like this at scale. Includes the causal variant "not because X, but because Y" where every explanation is framed as a surprise reveal, the em-dash dismissal "X -- not Y", and the cross-sentence reframe where the same noun is negated then repositioned: "The question isn't X. The question is Y."

**Avoid patterns like:**
- "It's not bold. It's backwards."
- "Feeding isn't nutrition. It's dialysis."
- "Half the bugs you chase aren't in your code. They're in your head."

### "Not X. Not Y. Just Z."

The dramatic countdown pattern. AI builds tension by negating two or more things before revealing the actual point. Creates a false sense of narrowing down to the truth.

**Avoid patterns like:**
- "Not a bug. Not a feature. A fundamental design flaw."
- "Not ten. Not fifty. Five hundred and twenty-three lint violations across 67 files."
- "not recklessly, not completely, but enough"

### "The X? A Y."

Self-posed rhetorical questions answered immediately in the next sentence or clause. The model asks a question nobody was asking, then answers it for dramatic effect. Thinks this is the epitome of great writing.

**Avoid patterns like:**
- "The result? Devastating."
- "The worst part? Nobody saw it coming."
- "The scary part? This attack vector is perfect for developers."

### Anaphora Abuse

Repeating the same sentence opening multiple times in quick succession.

**Avoid patterns like:**
- "They assume that users will pay... They assume that developers will build... They assume that ecosystems will emerge... They assume that..."
- "They could expose... They could offer... They could provide... They could create... They could let... They could unlock..."
- "They have built engines, but not vehicles. They have built power, but not leverage. They have built walls, but not doors."

### Tricolon Abuse

Overuse of the rule-of-three pattern, often extended to four or five. A single tricolon is elegant; three back-to-back tricolons are a pattern recognition failure.

**Avoid patterns like:**
- "Products impress people; platforms empower them. Products solve problems; platforms create worlds. Products scale linearly; platforms scale exponentially."
- "identity, payments, compute, distribution"
- "workflows, decisions, and interactions"

### "It's Worth Noting"

Filler transitions that signal nothing. AI uses these phrases to introduce new points without actually connecting them to the previous argument. Also includes: "It bears mentioning", "Importantly", "Interestingly", "Notably".

**Avoid patterns like:**
- "It's worth noting that this approach has limitations."
- "Importantly, we must consider the broader implications."
- "Interestingly, this pattern repeats across industries."

### Superficial Analyses

Tacking a present participle ("-ing") phrase onto the end of a sentence to inject shallow analysis that says nothing. The model attaches significance, legacy, or broader meaning to mundane facts using phrases like "highlighting its importance", "reflecting broader trends", or "contributing to the development of...".

**Avoid patterns like:**
- "contributing to the region's rich cultural heritage"
- "This etymology highlights the enduring legacy of the community's resistance and the transformative power of unity in shaping its identity."
- "underscoring its role as a dynamic hub of activity and culture"

### False Ranges

Using "from X to Y" constructions where X and Y aren't on any real scale. In legitimate use, "from X to Y" implies a spectrum with a meaningful middle. AI uses it as a fancy way to list two loosely related things. "From innovation to cultural transformation" -- what's in between???? Nothing!

**Avoid patterns like:**
- "From innovation to implementation to cultural transformation."
- "From the singularity of the Big Bang to the grand cosmic web."
- "From problem-solving and tool-making to scientific discovery, artistic expression, and technological innovation."

---

## Paragraph Structure

### Short Punchy Fragments

Excessive use of very short sentences or sentence fragments as standalone paragraphs for manufactured emphasis. RLHF training has pushed models toward "writing for readability" aimed at the lowest common denominator: one thought per sentence, no mental state-keeping required. It's an inhuman style. No real person writes first drafts this way because it doesn't match how humans think or speak.

**Avoid patterns like:**
- "He published this. Openly. In a book. As a priest."
- "These weren't just products. And the software side matched. Then it professionalised. But I adapted."
- "Platforms do."

### Listicle in a Trench Coat

Numbered or labeled points dressed up as continuous prose. The model writes what is essentially a listicle but wraps each point in a paragraph that starts with "The first... The second... The third..." to disguise the format. Perhaps you told it to stop generating lists and it decided to do this instead... still very common.

**Avoid patterns like:**
- "The first wall is the absence of a free, scoped API... The second wall is the lack of delegated access... The third wall is the absence of scoped permissions..."
- "The second takeaway is that... The third takeaway is that... The fourth takeaway is that..."

---

## Tone

### "Here's the Kicker"

False suspense transitions that promise a revelation but deliver a point that did NOT need the buildup. The model uses these phrases to manufacture drama before an otherwise unremarkable observation LOL. Also includes: "Here's the thing", "Here's where it gets interesting", "Here's what most people miss", "Here's the starting point", "Here's the deal".

**Avoid patterns like:**
- "Here's the kicker."
- "Here's the thing about AI adoption."
- "Here's where it gets interesting."

### "Think of It As..."

The patronizing analogy. AI constantly reaches for "Think of it as..." or "It's like a..." to simplify concepts. The model defaults to teacher mode and assumes the reader needs a metaphor to understand anything. Often produces analogies that are less clear than the original concept.

**Avoid patterns like:**
- "Think of it like a highway system for data."
- "Think of it as a Swiss Army knife for your workflow."
- "It's like asking someone to buy a car they're only allowed to sit in while it's parked."

### "Imagine a World Where..."

The classic AI invitation to futurism. To sell the argument usually begins with "Imagine" followed by a list of wonderful things that will happen if the reader agrees with the premise.

**Avoid patterns like:**
- "Imagine a world where every tool you use -- your calendar, your inbox, your documents, your CRM, your code editor -- has a quiet intelligence behind it..."
- "In that world, workflows stop being collections of manual steps and start becoming orchestrations."

### False Vulnerability

Simulated self-awareness or honesty that reads as performative. The model pretends to break the fourth wall or admit a bias, creating a false sense of authenticity. Real vulnerability is specific and uncomfortable; AI vulnerability is polished and risk-free!!!!

**Avoid patterns like:**
- "And yes, I'm openly in love with the platform model"
- "And yes, since we're being honest: I'm looking at you, OpenAI, Google, Anthropic, Meta"
- "This is not a rant; it's a diagnosis"

### "The Truth Is Simple"

Asserting that something is obvious, clear or simple instead of actually proving it. If you have to tell the reader your point is clear, it very likely isn't. Also includes the dramatic reveal variant: "but none of them is the real story. The real story is..." -- claiming privileged insight while waving away everything before it.

**Avoid patterns like:**
- "The reality is simpler and less flattering"
- "History is unambiguous on this point"
- "History is clear, the metrics are clear, the examples are clear"

### Grandiose Stakes Inflation

Everything is the most important thing ever. AI inflates the stakes of every argument to world-historical significance. A blog post about API pricing becomes a meditation on the fate of civilization.

**Avoid patterns like:**
- "This will fundamentally reshape how we think about everything."
- "will define the next era of computing"
- "something entirely new"

### "Let's Break This Down"

The pedagogical voice that assumes the reader needs hand-holding. AI defaults to a teacher-student dynamic even when writing for expert audiences. Also includes: "Let's unpack this", "Let's explore", "Let's dive in".

**Avoid patterns like:**
- "Let's break this down step by step."
- "Let's unpack what this really means."
- "Let's explore this idea further."

### Vague Attributions

Attributing claims to unnamed authorities instead of being specific. AI loves to invoke "experts", "observers", "industry reports", and "several publications" without naming anyone. It also inflates the quantity of sources -- presenting what one person said as a widely held view, or writing "several publications have cited" when it means two. If you can't name the expert, you don't have a source.

**Avoid patterns like:**
- "Experts argue that this approach has significant drawbacks."
- "Industry reports suggest that adoption is accelerating."
- "Observers have cited the initiative as a turning point."

### Invented Concept Labels

AI clusters invented compound labels that sound analytical without being grounded. It appends abstract problem-nouns (paradox, trap, creep, divide, vacuum, inversion) to domain words -- "supervision paradox", "acceleration trap", "workload creep" -- and uses them as if they're established, rigorously defined terms. They function as rhetorical shorthand: name a thing, skip the argument. Multiple such labels in the same piece is a strong signal of AI slop.

**Avoid patterns like:**
- "the supervision paradox"
- "the acceleration trap"
- "workload creep"

---

## Formatting

### Em-Dash Addiction

Compulsive overuse of em dashes for dramatic pauses, parenthetical asides and pivot points. A human writer might use 2-3 per piece (and naturally); AI will use 20+.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "The tinkerer spirit didn't die of natural causes -- it was bought out."
- "Not recklessly, not completely -- but enough -- enough to matter."

### Double-Hyphen Dash

The em dash wearing a false moustache. Once "em dash means AI" became common knowledge, the character started getting swapped for a double hyphen: sometimes because the text passed through a markdown conversion, sometimes because someone ran a find-and-replace to look more human, sometimes because the model was steered off the character while keeping the habit. Either way the compulsive mid-sentence pivot survives the substitution, which is what actually gives it away. Writers who reach for double hyphens honestly do so once or twice out of typographic laziness, rarely fifteen times in one post. Flagged at five or more per thousand words.

**Avoid patterns like:**
- "The problem -- and this is the part nobody talks about -- is systemic."
- "It's not a rewrite -- it's a reckoning."
- "We shipped it fast -- maybe too fast -- and paid for it later."

### Bold-First Bullets

Every bullet point or list item starts with a bolded phrase or sentence. Extremely common in Claude and ChatGPT markdown output. Almost nobody formats lists this way when writing by hand. It's a telltale sign of AI-generated documentation and blog posts AND README files (especially with emojis).

**Avoid patterns like:**
- "Every single bullet point begins with a bold keyword."
- "**Security**: Environment-based configuration with..."
- "**Performance**: Lazy loading of expensive resources..."

### Unicode Decoration

Use of unicode arrows (->), smart/curly quotes, and other special characters that can't be easily typed on a standard keyboard. Real writers typing in a text editor produce straight quotes and -> or =>. Claude in particular loves the -> arrow.

**Avoid patterns like:**
- "Input → Processing → Output"
- "This leads to better outcomes → which means higher engagement"
- "“Smart quotes” instead of straight "quotes" that you’d actually type"

---

## Composition

### Fractal Summaries

"What I'm going to tell you; what I'm telling you; what I just told you" -- applied at every level of the document. Every subsection gets a summary. Every section gets a summary. The document itself gets a summary.

**Avoid patterns like:**
- "In this section, we'll explore... [3000 words later] ...as we've seen in this section."
- "A conclusion that restates every point already made in the previous 3000 words"
- "And so we return to where we began."

### The Dead Metaphor

Latching onto a single metaphor and beating it into the ground across the entire thing. A human writer would introduce a metaphor, use it then move on. AI will repeat the same metaphor 5-10 times.

**Avoid patterns like:**
- "The ecosystem needs ecosystems to build ecosystem value."
- "Walls and doors used 30+ times in the same article"
- "Every paragraph finds a way to say "primitives" again"

### Historical Analogy Stacking

ESPECIALLY COMMON IN TECHNICAL WRITING: Rapid-fire listing of historical companies or tech revolutions to build false authority.

**Avoid patterns like:**
- "Apple didn't build Uber. Facebook didn't build Spotify. Stripe didn't build Shopify. AWS didn't build Airbnb."
- "Every major technological shift -- the web, mobile, social, cloud -- followed the same pattern."
- "Take Spotify... Or consider Uber... Airbnb followed a similar path... Shopify is another example... Even Discord..."

### One-Point Dilution

Making a single argument and restating it in 10 different ways across thousands of words. The model pads a simple thesis to feel "comprehensive" by rephrasing the same idea with different metaphors, examples, and framings. An 800-word argument becomes 4000 words of circular repetition.

**Avoid patterns like:**
- "The same point, restated eight ways across 4000 words."
- "Each section rephrases the thesis with a different metaphor but adds nothing new"

### Content Duplication

Repeating entire sections or paragraphs verbatim within the same piece. This happens when the model loses track of what it has already written, especially in longer pieces. A dead giveaway of unedited AI output. Less common nowadays.

**Avoid patterns like:**
- "The same section appeared twice, word-for-word identical."
- "Paragraph 3 and paragraph 17 are the same sentence reworded"

### The Signposted Conclusion

Explicitly announcing the conclusion with "In conclusion", "To sum up", or "In summary". Competent writing doesn't need to tell you it's concluding. The reader can feel it. AI signals its structural moves because it's following a template, not writing organically.

**Avoid patterns like:**
- "In conclusion, the future of AI depends on..."
- "To sum up, we've explored three key themes..."
- "In summary, the evidence suggests..."

### "Despite Its Challenges..."

The rigid formula where AI acknowledges problems only to immediately dismiss them. Always follows the same beat: "Despite its [positive words], [subject] faces challenges..." then ends with "Despite these challenges, [optimistic conclusion].".

**Avoid patterns like:**
- "Despite these challenges, the initiative continues to thrive."
- "Despite its industrial and residential prosperity, Korattur faces challenges typical of urban areas."
- "Despite their promising applications, pyroelectric materials face several challenges that must be addressed for broader adoption."

---

Remember: any of these patterns used once might be fine. The problem is when
multiple tropes appear together or when a single trope is used repeatedly.
Write like a human: varied, imperfect, specific.
