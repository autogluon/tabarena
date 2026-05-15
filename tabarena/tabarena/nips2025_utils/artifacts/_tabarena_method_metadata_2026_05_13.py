from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabpfnv3_method_metadata = MethodMetadata(
    method="TabPFN-3",
    method_type="config",
    display_name="TabPFN-3",
    compute="gpu",
    ag_key="TA-TABPFN-3",
    config_default="TabPFN-3_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    date="2026-05-13",
    reference_url="https://arxiv.org/abs/2605.13986",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    verified=True,
)


limix_metadata = MethodMetadata(
    method="LimiX",
    method_type="config",
    display_name="LimiX",
    compute="gpu",
    date="2026-05-13",
    ag_key="TA-LIMIX",
    config_default="LimiX_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=True,
    reference_url="https://arxiv.org/abs/2509.03505",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
)
