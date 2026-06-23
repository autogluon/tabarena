from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.orionmsp.hpo import gen_orionmsp
from tabarena.models.orionmsp.model import OrionMSPModel, prefetch_weights

orionmsp_method_metadata = MethodMetadata.config(
    method="OrionMSP",
    artifact_name="tabarena-2026-05-13",
    display_name="OrionMSP",
    compute="gpu",
    date="2026-05-13",
    ag_key="TA-ORION-MSP",
    config_default="OrionMSP_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    verified=False,
    s3_bucket="tabarena",
    s3_prefix="cache",
    cache_type="r2",
    reference_url="https://arxiv.org/abs/2511.02818",
)


orionmsp_info = ModelInfo(
    model_cls=OrionMSPModel,
    search_space=gen_orionmsp,
    method_metadata=orionmsp_method_metadata,
    pip_extra=("tabtune @ git+https://github.com/Lexsi-Labs/TabTune.git",),
    prefetch_weights=prefetch_weights,
)
