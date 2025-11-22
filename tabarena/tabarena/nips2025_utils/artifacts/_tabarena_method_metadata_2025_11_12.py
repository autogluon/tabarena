from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

realtabpfn25_metadata = MethodMetadata(
    method="RealTabPFN-v2.5",
    method_type="config",
    compute="gpu",
    date="2025-11-12",
    ag_key="REALTABPFN-V2.5",
    model_key="REALTABPFN-V2.5",
    config_default="RealTabPFN-v2.5_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-11-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2511.08667",
)
