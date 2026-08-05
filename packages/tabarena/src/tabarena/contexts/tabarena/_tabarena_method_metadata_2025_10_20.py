from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

_common_kwargs = dict(
    suite="tabarena-2025-10-20",
)

portfolio_metadata_paper_cr = MethodMetadata.tabarena_legacy_s3(
    method="Portfolio-N200-4h",
    method_type="portfolio",
    # A multi-model pipeline rather than a single model, so it groups with the systems.
    method_class="system",
    date="2025-10-20",
    has_raw=False,
    has_processed=False,
    **_common_kwargs,
)
