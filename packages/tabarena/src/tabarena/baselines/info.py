"""Baselines and portfolio entries — methods that aren't per-model wrappers.

These have `method_type="baseline"` or `method_type="portfolio"` (no
configurable model_cls / search_space), so they don't fit the
`tabarena.models.<key>.ModelInfo` shape used by the per-model registry.
They live here as standalone `MethodMetadata` instances.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

_s3_cache_kwargs = dict(
    s3_bucket="tabarena",
    s3_prefix="cache",
)


ag_130_metadata = MethodMetadata(
    method="AutoGluon_v130",
    name="AutoGluon 1.3 (best, 4h)",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="baseline",
    compute="cpu",
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    **_s3_cache_kwargs,
)


portfolio_metadata = MethodMetadata(
    method="Portfolio-N200-4h",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="portfolio",
    has_raw=False,
    has_processed=False,
    has_results=True,
    upload_as_public=True,
    **_s3_cache_kwargs,
)
