from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-10-20",
    cache_type="s3",
    name_suffix=None,
    # FIXME: technically LR and kNN are not verified
    verified=True,
)

portfolio_metadata_paper_cr = MethodMetadata.tabarena_public(
    method="Portfolio-N200-4h",
    method_type="portfolio",
    date="2025-10-20",
    has_raw=False,
    has_processed=False,
    **_common_kwargs,
)
