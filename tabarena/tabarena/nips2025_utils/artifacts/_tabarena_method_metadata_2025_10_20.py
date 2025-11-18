from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-10-20",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
)

lr_metadata = MethodMetadata(
    method="LinearModel",
    method_type="config",
    compute="cpu",
    date="2025-10-20",
    ag_key="LR",
    config_default="LinearModel_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    **_common_kwargs,
)
knn_metadata = MethodMetadata(
    method="KNeighbors",
    method_type="config",
    compute="cpu",
    date="2025-10-20",
    ag_key="KNN",
    config_default="KNeighbors_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    **_common_kwargs,
)
tabdpt_metadata = MethodMetadata(
    method="TabDPT_GPU",
    method_type="config",
    compute="gpu",
    date="2025-10-20",
    ag_key="TABDPT",
    model_key="TABDPT_GPU",
    config_default="TabDPT_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    **_common_kwargs,
)
portfolio_metadata_paper_cr = MethodMetadata(
    method="Portfolio-N200-4h",
    method_type="portfolio",
    date="2025-10-20",
    has_raw=False,
    has_processed=False,
    **_common_kwargs,
)
