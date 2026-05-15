from tabarena.models.perpetual_booster.info import perpetual_booster_method_metadata
from tabarena.models.tabstar.info import tabstar_method_metadata
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

# Legacy aliases — preserved for any external code still importing the older names.
perpetualbooster_metadata = perpetual_booster_method_metadata
tabstar_metadata = tabstar_method_metadata

tabpfn26_metadata = MethodMetadata(
    method="TabPFN-v2.6",
    method_type="config",
    display_name="TabPFN-2.6",
    compute="gpu",
    date="2026-03-25",
    ag_key="TABPFN-V2.6",
    model_key="TABPFN-V2.6",
    config_default="TabPFN-v2.6_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-03-18",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2511.08667",
    cache_type="r2",
)
