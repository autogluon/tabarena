from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabprep_gbm_metadata = MethodMetadata(
    method="PrepLightGBM",
    artifact_name="tabarena-2026-01-23",
    display_name="PrepLightGBM",
    method_type="config",
    compute="cpu",
    date="2026-01-23",
    ag_key="PREP_GBM",
    model_key="PREP_GBM",
    # config_default="PrepLightGBM_c1_BAG_L1",  # FIXME
    config_default="prep_LightGBM_icml_v3_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=True,
    is_bag=True,
    s3_bucket="tabarena",
    s3_prefix="cache",
    verified=True,
)

tabprep_lr_metadata = MethodMetadata(
    method="PrepLinearModel",
    artifact_name="tabarena-2026-01-23",
    display_name="PrepLinear",
    method_type="config",
    compute="cpu",
    date="2026-01-23",
    ag_key="PREP_LR",
    model_key="PREP_LR",
    # config_default="PrepLinearModel_c1_BAG_L1",  # FIXME
    config_default="prep_LinearModel_icml_v3_c1_BAG_L1",
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=True,
    is_bag=True,
    s3_bucket="tabarena",
    s3_prefix="cache",
    verified=True,
)


tabprep_tabm_metadata = MethodMetadata(
    method="PrepTabM",
    artifact_name="tabarena-2026-01-23",
    display_name="PrepTabM",
    method_type="config",
    compute="gpu",
    date="2026-01-23",
    ag_key="PREP_TABM",
    model_key="PREP_TABM",
    config_default="prep_TabM_c1_BAG_L1",  # FIXME
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=True,
    is_bag=True,
    s3_bucket="tabarena",
    s3_prefix="cache",
    verified=True,
)


tabprep_realtabpfnv250_metadata = MethodMetadata(
    method="PrepRealTabPFN-v2.5",
    artifact_name="tabarena-2026-01-23",
    display_name="PrepRealTabPFN-2.5",
    method_type="config",
    compute="gpu",
    date="2026-01-23",
    ag_key="PREP_REALTABPFN-V2.5",
    model_key="PREP_REALTABPFN-V2.5",
    config_default="prep_RealTabPFN-v2.5_c1_BAG_L1",  # FIXME
    name_suffix=None,
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    can_hpo=True,
    is_bag=False,
    s3_bucket="tabarena",
    s3_prefix="cache",
    verified=True,
)
