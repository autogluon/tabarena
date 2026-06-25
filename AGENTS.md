# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs live in `README.md`.

## Project Overview

TabArena is a living benchmark for tabular machine learning. It evaluates ML methods across 51 curated datasets with cross-validated ensembles, HPO simulation, and leaderboard generation. Built on top of AutoGluon.

## Repository Layout

This repo is a **uv workspace** (root `pyproject.toml`). Its installable packages live under `packages/` (workspace members `tabarena`, `bencheval`, `tabflow_slurm`), alongside supporting dirs:

- `packages/tabarena/` — Core package. Repository pattern for benchmark data, model wrappers, simulation, evaluation, plotting. Depends on AutoGluon and `bencheval`.
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
- **`ExperimentRunner` / `ExperimentBatchRunner`** (`packages/tabarena/src/tabarena/benchmark/experiment/`) — Execute model fitting across tasks. Configured via YAML (`experiment_constructor.py`).
- **`ZeroshotSimulatorContext`** (`packages/tabarena/src/tabarena/simulation/`) — Manages config rankings for HPO simulation and portfolio generation.
- **`BenchmarkEvaluator`** (`packages/bencheval/src/bencheval/evaluator.py`) — Leaderboard computation from results DataFrames. Independent of the core `tabarena` package.

### Data caching

TabArena uses four independent caches. Configure them all at once with `tabarena.caching.CacheConfig` — the single, documented surface (`TabArenaContext(cache_config=CacheConfig.from_root(...))`; the context applies it on construction and re-applies it inside `run_jobs`, so distributed workers inherit it). The SLURM path uses the **same** object: the setup embeds `context.cache_config` in the `JobBatch`, and each worker applies it (no `--openml_cache_dir` wiring). See the `CacheConfig` docstring for the authoritative per-cache reference.

| Cache | `CacheConfig` field | Holds | Set via | Default |
|---|---|---|---|---|
| OpenML (most important) | `openml` | Materialized datasets + CV splits + all TabArena-derived task artifacts (`tabarena_tasks/`, `tabarena_text_cache/`, `tabarena_metadata_cache/`, `local/datasets/`) | `openml.config.set_root_cache_directory` (no env var) | `~/.cache/openml` |
| HuggingFace | `huggingface` | Foundation-model weights **and** the one-time raw dataset download (data_foundry/BeyondArena), later materialized into the OpenML cache | `HF_HOME` | `~/.cache/huggingface/hub` |
| TabArena | `tabarena` | Results / baselines / leaderboard artifacts (~100 GB raw, ~10 GB processed, <1 MB results per method) | `set_tabarena_cache_root` / `TABARENA_CACHE` | `~/.cache/tabarena` |
| Results (run output) | `results` | The runner's `expname` (`{expname}/data/{method}/{task}/{repeat}_{fold}/results.pkl`) | `run_jobs(expname=...)` | throwaway temp dir |

## Conventions

- **Add a new model**: create one folder `packages/tabarena/src/tabarena/models/<model>/` (`model.py`, `hpo.py`, `info.py`, `__init__.py`), then edit `models/__init__.py` (lazy class entry), `models/utils.py` (name→generator map), and `packages/tabarena/pyproject.toml` (a per-model extra). The registry auto-discovers the model from its `info.py`, and `tests/tabarena/models/test_all_models.py` then fits it automatically — there is **no per-model test file**. Only add an entry to `tests/tabarena/models/smoke_configs.py` if the smoke fit needs faster toy hyperparameters or a restricted problem-type set (keyed by the model's `MethodMetadata.method`). **Use the `add-model` skill**, which encodes this and points to reference implementations (foundation / torch / sklearn).
- **Imports**: `from __future__ import annotations` must be the first import in every `.py` file. Use absolute imports rooted at the package (e.g., `from tabarena.repository import EvaluationRepository`).
- **Optional dependencies**: each model has its own pyproject extra under `packages/tabarena/pyproject.toml`; the `benchmark` extra is the union. Heavy/optional libs must never be imported at module top-level in core paths — import inside the model wrapper.
- **Scoping jobs**: when scoping `context.build_jobs` / `build_and_run_jobs` (in examples and code), pass a typed `TaskSubset` via `task_subset=` — it is the single source of truth for the available filters (`subset`, `dataset_names`, `split_indices`, `problem_types`, `n_train_samples`, ...). Prefer it over the loose `dataset_names=` / `build_kwargs={...}` keyword conveniences.
- **No new top-level docs files** unless the user asks. Edit existing files in place.

## PR & Commit Guidance

- Keep commits focused; do not bundle unrelated refactors with bug fixes.
- Run `ruff check .` and `pytest` on affected paths before opening a PR.
- CI is mandatory on `main` PRs.

## Things to Avoid

- Do not add a `tst/` dir or per-package `tests/` dirs — all tests live in the single top-level `tests/`, grouped by package (`tests/tabarena/`, `tests/bencheval/`, `tests/tabflow_slurm/`, `tests/integration/`).
- Do not import optional model dependencies at the top of shared modules; lazy-import inside the wrapper.
- Do not skip `from __future__ import annotations` — ruff will fail CI.
- Do not change the public API of `EvaluationRepository`, `TabularModelPredictions`, or `bencheval.evaluator.BenchmarkEvaluator` without explicit user direction; they are consumed by external scripts and artifacts.
