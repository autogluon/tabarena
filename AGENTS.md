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
- `tst/` — Tests (note: `tst/`, **not** `tests/`).

## Setup Commands

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/). This is a uv *virtual workspace*
(the root `pyproject.toml` has no `[project]` table), so install the `tabarena` package directly
from `packages/tabarena` rather than running `uv sync` at the root. `--prerelease=allow` is required
for the pre-release AutoGluon dependency. From the repo root, after creating/activating a venv
(`uv venv --seed --python 3.12 && source .venv/bin/activate`):

```bash
uv pip install --prerelease=allow -e "./packages/tabarena"               # Evaluation-only (leaderboard/metrics)
uv pip install --prerelease=allow -e "./packages/tabarena[benchmark]"    # Full install including model fitting
```

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

```bash
pytest                                    # All tests
pytest tst/test_metrics.py                # Single file
pytest tst/test_metrics.py::test_name -x  # Single test, stop on failure
```

CI (`.github/workflows/`) runs `pytest` on Python 3.11 against an editable install of `./packages/tabarena`.

**Avoid running `tst/models/` in bulk** unless your change would likely impact a specific model: these tests fit real models (slow, and some need GPUs/licenses/network), and they are only affected by changes to the model files under `packages/tabarena/src/tabarena/models/<model>/`. When you touch a model, run just its test (`pytest tst/models/test_<model>.py`).

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
- **`TabArena`** (`packages/bencheval/src/bencheval/tabarena.py`) — Leaderboard computation from results DataFrames. Independent of the core `tabarena` package.

### Data caching

Artifacts download to `~/.cache/tabarena/` by default; override with `TABARENA_CACHE`. Three tiers per method: raw data (~100 GB), processed predictions (~10 GB), results DataFrames (<1 MB).

## Conventions

- **Add a new model**: create one folder `packages/tabarena/src/tabarena/models/<model>/` (`model.py`, `hpo.py`, `info.py`, `__init__.py`), then edit `models/__init__.py` (lazy class entry), `models/utils.py` (name→generator map), and `packages/tabarena/pyproject.toml` (a per-model extra), plus a `tst/models/test_<model>.py`. The registry auto-discovers the model from its `info.py` — no manual registry edit. **Use the `add-model` skill**, which encodes this and points to reference implementations (foundation / torch / sklearn).
- **Imports**: `from __future__ import annotations` must be the first import in every `.py` file. Use absolute imports rooted at the package (e.g., `from tabarena.repository import EvaluationRepository`).
- **Optional dependencies**: each model has its own pyproject extra under `packages/tabarena/pyproject.toml`; the `benchmark` extra is the union. Heavy/optional libs must never be imported at module top-level in core paths — import inside the model wrapper.
- **No new top-level docs files** unless the user asks. Edit existing files in place.

## PR & Commit Guidance

- Keep commits focused; do not bundle unrelated refactors with bug fixes.
- Run `ruff check .` and `pytest` on affected paths before opening a PR.
- CI is mandatory on `main` PRs.

## Things to Avoid

- Do not add `tests/` — use `tst/`.
- Do not import optional model dependencies at the top of shared modules; lazy-import inside the wrapper.
- Do not skip `from __future__ import annotations` — ruff will fail CI.
- Do not change the public API of `EvaluationRepository`, `TabularModelPredictions`, or `bencheval.tabarena.TabArena` without explicit user direction; they are consumed by external scripts and artifacts.
