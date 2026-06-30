# CLAUDE.md

The primary agent guide for this repo is [`AGENTS.md`](./AGENTS.md). **Read it first** — everything in it applies here.

This file only documents Claude-specific extensions.

## Claude-Specific Tooling

### Skills (`.claude/skills/`)

- **`add-model`** — Use whenever the user asks to add/integrate/wrap a new tabular ML model. It encodes the full change: a per-model folder (`model.py` wrapper, `hpo.py` search space, `info.py` registry entry) plus edits to `models/__init__.py`, `models/utils.py`, and the `pyproject.toml` extra — and points to reference implementations for each model class (foundation, torch, sklearn-like). The model is auto-discovered from its `info.py` and fit-tested automatically by `tests/tabarena/models/test_all_models.py` (no per-model test file); only add a `smoke_configs.py` override if its toy fit needs faster hyperparameters.
- **`upload-method`** — Use whenever a maintainer points at a benchmark run's output dir and wants to process / upload / register a method's results (e.g. "upload this method", "host/publish `<model>`'s results", "register `<model>` in the leaderboard"). It builds the command sheet the maintainer runs (`scripts/run_process_method.py` inspect → `--process`; `scripts/run_upload_results.py` dry-run → `--no-dry-run`) and does the edits Claude can land now: the model's `info.py` `MethodMetadata` (suite / date / `cache_type` / `cache_kwargs` / verified) and the arena-collection registration in `contexts/<arena>/methods.py`. Mirrors AGENTS.md → "Processing & uploading method artifacts (maintainers)".

When the user describes work that matches a skill's trigger criteria, invoke the skill via the Skill tool instead of recreating the steps manually.

## Working Style in This Repo

- **Always run `ruff check --fix`** on touched files before reporting a task complete. The `from __future__ import annotations` requirement (isort `required-imports`) is the most common CI failure on new files. CI now enforces `ruff check .` + `ruff format --check .`, and a pinned `.pre-commit-config.yaml` is available (`pre-commit install`) — see AGENTS.md "Lint & Format".
- **Tests live in the top-level `tests/`, grouped by package** — when adding a test for `packages/tabarena/src/tabarena/<area>/foo.py`, mirror the path under `tests/tabarena/<area>/test_foo.py`. (`tests/bencheval/` and `tests/tabflow_slurm/` hold the other two packages; `tests/integration/` is for cross-package tests.)
- **Model-fit tests are deselected by default and consolidated** — `tests/tabarena/models/test_all_models.py` fits every registered model (marked `models`, skipped by the default `pytest`). Run one with `pytest -m models -k <Model>`; they need `tabarena[benchmark]`, are slow, and skip GPU-only models without CUDA. There is no per-model test file. For non-model changes, the default `pytest` already skips them.
- **Optional model deps**: when adding code that imports an optional library (e.g., `tabpfn`, `tabicl`, `xrfm`), keep the import inside the wrapper's `_fit` / class body, never at module top-level — top-level imports break installs without that extra.
- **Docstrings describe current behavior, not refactor history.** When you rename, move, or remove code, write docstrings/comments about what the code *is* now — don't leave `Formerly X`, `used to live in Y`, `previously did Z`, or `now delegates here` narratives, or dead references to deleted classes/modules. Rename/removal history belongs in git and the PR, not the source. Keep a historical note only when it's genuinely important to understanding the current code (e.g. a non-obvious backwards-compat shim). (`previously` / `used to live` are fine when they explain *current* behavior and reference code that still exists.)