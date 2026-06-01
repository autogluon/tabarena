# CLAUDE.md

The primary agent guide for this repo is [`AGENTS.md`](./AGENTS.md). **Read it first** — everything in it applies here.

This file only documents Claude-specific extensions.

## Claude-Specific Tooling

### Skills (`.claude/skills/`)

- **`add-model`** — Use whenever the user asks to add/integrate/wrap a new tabular ML model. It encodes the full change: a per-model folder (`model.py` wrapper, `hpo.py` search space, `info.py` registry entry) plus edits to `models/__init__.py`, `models/utils.py`, and the `pyproject.toml` extra, and a test — and points to reference implementations for each model class (foundation, torch, sklearn-like). The model is auto-discovered from its `info.py`.

When the user describes work that matches a skill's trigger criteria, invoke the skill via the Skill tool instead of recreating the steps manually.

## Working Style in This Repo

- **Always run `ruff check --fix`** on touched files before reporting a task complete. The `from __future__ import annotations` requirement (isort `required-imports`) is the most common CI failure on new files.
- **Tests live in `tst/`** — when adding a test for `tabarena/tabarena/<area>/foo.py`, mirror the path under `tst/<area>/test_foo.py`.
- **Optional model deps**: when adding code that imports an optional library (e.g., `tabpfn`, `tabicl`, `xrfm`), keep the import inside the wrapper's `_fit` / class body, never at module top-level — top-level imports break installs without that extra.