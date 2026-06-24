from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.xrfm.hpo import gen_xrfm
from tabarena.models.xrfm.model import XRFMModel

xrfm_method_metadata = MethodMetadata.config(
    method="xRFM_GPU",
    artifact_name="tabarena-2025-09-03",
    ag_key="XRFM",
    model_key="XRFM_GPU",
    config_default="xRFM_GPU_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-09-03",
    reference_url="https://arxiv.org/abs/2508.10053",
    display_name="xRFM",
)


xrfm_info = ModelInfo(
    model_cls=XRFMModel,
    search_space=gen_xrfm,
    method_metadata=xrfm_method_metadata,
    pip_extra=("xrfm[cu12]",),
)
