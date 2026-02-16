from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

tabiclv2_metadata = MethodMetadata(
    method="TabICLv2",
    method_type="config",
    display_name="TabICLv2",
    compute="gpu",
    date="2026-02-16",
    ag_key="TABICLV2",
    model_key="TABICLV2",
    config_default="TabICLv2_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-02-16",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2602.11139",
)
