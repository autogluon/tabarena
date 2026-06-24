from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.sap_rpt_oss.hpo import gen_sap_rpt_oss
from tabarena.models.sap_rpt_oss.model import SAPRPTOSSModel, prefetch_weights

sap_rpt_oss_method_metadata = MethodMetadata.config(
    method="SAP-RPT-OSS",
    suite="tabarena-2025-11-25",
    ag_key="SAP-RPT-OSS",
    config_default="SAP-RPT-OSS_c1_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-11-25",
    reference_url="https://arxiv.org/abs/2506.10707",
    display_name="SAP-RPT-OSS",
    verified=False,
)


sap_rpt_oss_info = ModelInfo(
    model_cls=SAPRPTOSSModel,
    search_space=gen_sap_rpt_oss,
    method_metadata=sap_rpt_oss_method_metadata,
    pip_extra=(
        "sap_rpt_oss @ git+https://github.com/SAP-samples/sap-rpt-1-oss.git@a323a0aff976fda4ac43c3196a92406de7689aaa",
    ),
    prefetch_weights=prefetch_weights,
)
