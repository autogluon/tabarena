from __future__ import annotations

from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabm.hpo import gen_tabm
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabm_method_metadata = MethodMetadata(
    method="TabM",
    method_type="config",
    display_name="TabM (CPU)",
    compute="cpu",
    date="2025-06-12",
    ag_key="TABM",
    config_default="TabM_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2410.24210",
)


tabm_gpu_method_metadata = MethodMetadata(
    method="TabM_GPU",
    method_type="config",
    display_name="TabM",
    compute="gpu",
    date="2025-06-12",
    ag_key="TABM",
    model_key="TABM",
    config_default="TabM_GPU_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix="_GPU",
    verified=True,
    reference_url="https://arxiv.org/abs/2410.24210",
)


tabm_info = ModelInfo(
    model_cls=TabMModel,
    search_space=gen_tabm,
    method_metadata=tabm_method_metadata,
    pip_extra=("torch",),
)


tabm_gpu_info = ModelInfo(
    model_cls=TabMModel,
    search_space=gen_tabm,
    method_metadata=tabm_gpu_method_metadata,
    pip_extra=("torch",),
)
