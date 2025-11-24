from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


# LightGBM w/ custom preprocessing pipeline (only first 3 repeats)
# s3 cache = "cache_aio"
gbm_aio_0808_metadata = MethodMetadata(
    method="LightGBM_aio_0808",
    artifact_name="LightGBM_aio_0808",
    method_type="config",
    compute="cpu",
    date="2025-08-08",
    ag_key="GBM",
    model_key="GBM_aio_0808",
    config_default="LightGBM_aio_0808_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=True,
    is_bag=True,
    s3_bucket="tabarena",
    s3_prefix="cache_aio",
    verified=False,
)
