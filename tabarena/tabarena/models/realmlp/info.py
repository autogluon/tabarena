from __future__ import annotations

from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.realmlp.hpo import gen_realmlp
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


realmlp_method_metadata = MethodMetadata(
    method="RealMLP_GPU",
    method_type="config",
    display_name="RealMLP",
    compute="gpu",
    date="2025-09-03",
    ag_key="TA-REALMLP",
    model_key="REALMLP_GPU",
    config_default="RealMLP_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    verified=True,
    reference_url="https://arxiv.org/abs/2407.04491",
    artifact_name="tabarena-2025-09-03",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)


realmlp_info = ModelInfo(
    model_cls=RealMLPModel,
    search_space=gen_realmlp,
    method_metadata=realmlp_method_metadata,
)
