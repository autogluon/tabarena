from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

realtabpfn25_metadata = MethodMetadata(
    method="RealTabPFN-v2.5",
    method_type="config",
    display_name="RealTabPFN-2.5",
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

contexttab_metadata = MethodMetadata(
    method="SAP-RPT-OSS",
    method_type="config",
    display_name="SAP-RPT-OSS",
    compute="gpu",
    date="2025-11-25",
    ag_key="SAP-RPT-OSS",
    model_key="SAP-RPT-OSS",
    config_default="SAP-RPT-OSS_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-11-25",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=False,
    reference_url="https://arxiv.org/abs/2506.10707",
)

# TODO: finalize metadata
tabstar = MethodMetadata(
    method="TabSTAR",
    method_type="config",
    compute="gpu",
    date="2025-12-13",
    ag_key="TABSTAR",
    model_key="TABSTAR",
    config_default="TabStar_c1_BAG_L1",
    can_hpo=False, # TODO: add results with HPO
    is_bag=True,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-XXXX", # TODO
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True, # TODO: get confirmation
    reference_url="https://arxiv.org/abs/2505.18125",
)
