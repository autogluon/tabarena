"""Evaluation runners for benchmark results.

Two flavors share the post-process/cache/subset-loop skeleton (in ``_eval_common``):

* :func:`~tabarena.evaluation.benchmark_eval.run_eval` — TabArena v0.1, compared against the
  published paper baselines.
* :func:`~tabarena.evaluation.beyond_arena_eval.run_beyond_arena_eval` — BeyondArena (data-foundry),
  combining one or more runs and filtering by data-foundry subset predicates.

Only light dataclasses/functions are imported at module load here; the heavy result-processing /
comparison imports are deferred to inside the runners (keeps ``import tabarena.evaluation`` cheap
and avoids the historical import cycle through ``nips2025_utils``).
"""

from __future__ import annotations

from tabarena.evaluation.benchmark_eval import EvalMethod, TabArenaEvalConfig, run_eval
from tabarena.evaluation.beyond_arena_eval import (
    BenchmarkRun,
    BeyondArenaEvalConfig,
    run_beyond_arena_eval,
)

__all__ = [
    "BenchmarkRun",
    "BeyondArenaEvalConfig",
    "EvalMethod",
    "TabArenaEvalConfig",
    "run_beyond_arena_eval",
    "run_eval",
]
