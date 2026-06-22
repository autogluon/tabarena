"""Baselines and portfolio entries — methods that aren't per-model wrappers.

These have `method_type="baseline"` or `method_type="portfolio"` (no
configurable model_cls / search_space), so they don't fit the
`tabarena.models.<key>.ModelInfo` shape used by the per-model registry.
They live here as standalone `MethodMetadata` instances.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

ag_130_metadata = MethodMetadata.tabarena_public(
    method="AutoGluon_v130",
    name="AutoGluon 1.3 (best, 4h)",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="baseline",
    compute="cpu",
    cache_type="s3",
)


portfolio_metadata = MethodMetadata.tabarena_public(
    method="Portfolio-N200-4h",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="portfolio",
    has_raw=False,
    has_processed=False,
    cache_type="s3",
)
