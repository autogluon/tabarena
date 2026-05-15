from __future__ import annotations

from tabarena.benchmark.models.ag.xrfm.xrfm_model import XRFMModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.xrfm.hpo import gen_xrfm
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


xrfm_method_metadata = MethodMetadata(
    method="xRFM_GPU",
    method_type="config",
    display_name="xRFM",
    compute="gpu",
    date="2025-09-03",
    ag_key="XRFM",
    model_key="XRFM_GPU",
    config_default="xRFM_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    verified=True,
    reference_url="https://arxiv.org/abs/2508.10053",
    artifact_name="tabarena-2025-09-03",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)


xrfm_info = ModelInfo(
    model_cls=XRFMModel,
    search_space=gen_xrfm,
    method_metadata=xrfm_method_metadata,
    pip_extra=("xrfm[cu12]",),
)
