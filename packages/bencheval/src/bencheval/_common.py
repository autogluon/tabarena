from __future__ import annotations

# Column names produced by the evaluator and shared across its concern modules
# (validation, analysis, plotting). Defined here to avoid an evaluator <-> mixin
# import cycle. Re-exported from ``bencheval.evaluator`` for backward compatibility.
RANK = "rank"
IMPROVABILITY = "improvability"
LOSS_RESCALED = "loss_rescaled"
MINMAX_NORMALIZED_SCORE = "minmax_normalized_score"
FRONTIER_ADVANTAGE = "frontier_advantage"
