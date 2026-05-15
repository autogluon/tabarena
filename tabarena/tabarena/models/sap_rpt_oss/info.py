from __future__ import annotations

from tabarena.benchmark.models.ag.sap_rpt_oss.sap_rpt_oss_model import SAPRPTOSSModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.sap_rpt_oss.hpo import gen_sap_rpt_oss
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


sap_rpt_oss_method_metadata = MethodMetadata(
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


sap_rpt_oss_info = ModelInfo(
    model_cls=SAPRPTOSSModel,
    search_space=gen_sap_rpt_oss,
    method_metadata=sap_rpt_oss_method_metadata,
    pip_extra=(
        "sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@a323a0aff976fda4ac43c3196a92406de7689aaa",
    ),
)
