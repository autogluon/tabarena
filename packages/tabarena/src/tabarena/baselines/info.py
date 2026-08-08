"""Portfolio entries: methods that are neither per-model wrappers nor external systems.

A portfolio has `method_type="portfolio"` (no configurable model_cls / search_space), so it
does not fit the `tabarena.models.<key>.ModelInfo` shape used by the per-model registry. It
does carry `method_class="system"`: like AutoGluon it is a multi-model pipeline rather than a
single model, and the leaderboard groups it with the systems.

Benchmarked systems with a wrapper of their own live in `tabarena.systems`; the AutoGluon
entries that used to sit here are in `tabarena.systems.autogluon.info`.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems.autogluon.info import ag_130_metadata

__all__ = ["ag_130_metadata", "portfolio_metadata"]

portfolio_metadata = MethodMetadata.tabarena_legacy_s3(
    method="Portfolio-N200-4h",
    suite="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="portfolio",
    method_class="system",
    has_raw=False,
    has_processed=False,
)
