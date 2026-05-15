# AGENTS.md

Guidance for coding agents working in this repository. Human-facing docs live in `README.md`.

## Project Overview

TabArena is a living benchmark for tabular machine learning. It evaluates ML methods across 51 curated datasets with cross-validated ensembles, HPO simulation, and leaderboard generation. Built on top of AutoGluon.

## Repository Layout

This repo is a **uv workspace** (root `pyproject.toml`) with three packages plus standalone tooling:

- `tabarena/` — Core package. Repository pattern for benchmark data, model wrappers, simulation, evaluation, plotting. Depends on AutoGluon and `bencheval`.
- `bencheval/` — Standalone lightweight metrics/leaderboard package (ELO, win-rates, ranks, improvability). Computes leaderboards from results DataFrames. No dependency on `tabarena`.
- `tabflow/` — AWS SageMaker workflow orchestration. CLI entry points: `tabflow` (launch jobs) and `tabflow-download` (download results). Depends on `tabarena`.
- `tabflow_slurm/` — Standalone scripts (not a package) for running experiments on SLURM clusters.
- `examples/` — Usage examples for benchmarking, plotting, meta-learning, custom models.
- `tst/` — Tests (note: `tst/`, **not** `tests/`).

## Setup Commands

Requires Python 3.11–3.13 and [uv](https://docs.astral.sh/uv/).

```bash
uv sync                    # Evaluation-only install (leaderboard/metrics)
uv sync --extra benchmark  # Full install including model fitting
```

For editable AutoGluon development (one directory up):

```bash
../autogluon/full_install.sh
uv pip install --prerelease=allow -e "./tabarena[benchmark]"
```

## Lint & Format

```bash
ruff check .               # Lint (config: ruff.toml)
ruff format .              # Format
```

Key rules: `from __future__ import annotations` is required in every file (enforced via isort `required-imports`). Line length 120. Google-style docstrings. Run ruff on touched files before finishing a task.

## Testing

```bash
pytest                                    # All tests
pytest tst/test_metrics.py                # Single file
pytest tst/test_metrics.py::test_name -x  # Single test, stop on failure
```

CI (`.github/workflows/`) runs `pytest` on Python 3.11 against an editable install of `./tabarena`.

## Architecture

### Core data flow

```
Raw predictions → EvaluationRepository → Simulation/Portfolio → Results DataFrames → TabArena leaderboard (bencheval)
```

### Key abstractions

- **`EvaluationRepository`** (`tabarena/tabarena/repository/evaluation_repository.py`) — Central class combining config metadata/rankings (`ZeroshotSimulatorContext`), cached val/test predictions (`TabularModelPredictions`), and `GroundTruth`. Supports subsetting by datasets/folds/configs/problem_types and ensemble selection via mixins.
- **`TabularModelPredictions`** (`tabarena/tabarena/predictions/`) — Abstract base for prediction storage. Implementations: `TabularPredictionsInMemory` (dict-based) and `TabularPredictionsMemmap` (disk-based memory-mapped for large benchmarks). Structure: `{dataset: {fold: {val/test: {config: predictions}}}}`.
- **`AbstractExecModel`** (`tabarena/tabarena/benchmark/models/wrapper/abstract_class.py`) — Base for benchmarked models. AutoGluon model wrappers live under `tabarena/tabarena/benchmark/models/ag/<model>/`; matching HPO search-space generators live under `tabarena/tabarena/models/<model>/generate.py`. Registry: `tabarena/tabarena/benchmark/models/model_registry.py`.
- **`ExperimentRunner` / `ExperimentBatchRunner`** (`tabarena/tabarena/benchmark/experiment/`) — Execute model fitting across tasks. Configured via YAML (`experiment_constructor.py`).
- **`ZeroshotSimulatorContext`** (`tabarena/tabarena/simulation/`) — Manages config rankings for HPO simulation and portfolio generation.
- **`TabArena`** (`bencheval/bencheval/tabarena.py`) — Leaderboard computation from results DataFrames. Independent of the core `tabarena` package.

### Data caching

Artifacts download to `~/.cache/tabarena/` by default; override with `TABARENA_CACHE`. Three tiers per method: raw data (~100 GB), processed predictions (~10 GB), results DataFrames (<1 MB).

## Conventions

- **Add a new model**: touches ~7 locations (AG wrapper, search-space generator, two `__init__.py`s, `model_registry.py`, `models/utils.py` import map, `tabarena/pyproject.toml` extras, and a test). Use existing models in `tabarena/tabarena/benchmark/models/ag/` as templates — pick a structural neighbor (foundation/torch model vs. CPU/sklearn model).
- **Imports**: `from __future__ import annotations` must be the first import in every `.py` file. Use absolute imports rooted at the package (e.g., `from tabarena.repository import EvaluationRepository`).
- **Optional dependencies**: each model has its own pyproject extra under `tabarena/pyproject.toml`; the `benchmark` extra is the union. Heavy/optional libs must never be imported at module top-level in core paths — import inside the model wrapper.
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
