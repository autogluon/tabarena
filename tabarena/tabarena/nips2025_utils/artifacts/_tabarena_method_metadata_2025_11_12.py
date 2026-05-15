from tabarena.models.sap_rpt_oss.info import sap_rpt_oss_method_metadata
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

# Legacy alias — preserved for any external code still importing the older name.
contexttab_metadata = sap_rpt_oss_method_metadata

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

