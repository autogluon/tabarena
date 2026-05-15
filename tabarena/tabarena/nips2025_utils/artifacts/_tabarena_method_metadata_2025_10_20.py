from __future__ import annotations

from tabarena.models.knn.info import knn_method_metadata
from tabarena.models.lr.info import lr_method_metadata
from tabarena.models.tabdpt.info import tabdpt_method_metadata
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

# Legacy aliases — preserved for any external code still importing the older names.
knn_metadata = knn_method_metadata
lr_metadata = lr_method_metadata
tabdpt_metadata = tabdpt_method_metadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-10-20",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    # FIXME: technically LR and kNN are not verified
    verified=True,
)

portfolio_metadata_paper_cr = MethodMetadata(
    method="Portfolio-N200-4h",
    method_type="portfolio",
    date="2025-10-20",
    has_raw=False,
    has_processed=False,
    **_common_kwargs,
)
