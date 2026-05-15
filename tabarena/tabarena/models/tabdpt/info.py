from __future__ import annotations

from tabarena.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabdpt.hpo import gen_tabdpt
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabdpt_method_metadata = MethodMetadata(
    method="TabDPT_GPU",
    method_type="config",
    display_name="TabDPT",
    compute="gpu",
    date="2025-10-20",
    ag_key="TABDPT",
    model_key="TABDPT_GPU",
    config_default="TabDPT_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    reference_url="https://arxiv.org/abs/2410.18164",
    artifact_name="tabarena-2025-10-20",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
)


tabdpt_info = ModelInfo(
    model_cls=TabDPTModel,
    search_space=gen_tabdpt,
    method_metadata=tabdpt_method_metadata,
    pip_extra=("tabdpt>=1.1.10",),
)
