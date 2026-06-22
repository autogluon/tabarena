from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabm.hpo import gen_tabm
from tabarena.models.tabm.model import TabMModel

tabm_descriptor = ModelDescriptor(
    display_name="TabM",
    compute="gpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/2410.24210",
)

# CPU variant — same model class, same search space, different compute target.
tabm_method_metadata = tabm_descriptor.method_metadata(
    method="TabM",
    display_name="TabM (CPU)",
    compute="cpu",
    method_type="config",
    date="2025-06-12",
    ag_key="TABM",
    config_default="TabM_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    verified=True,
)


tabm_gpu_method_metadata = tabm_descriptor.method_metadata(
    method="TabM_GPU",
    method_type="config",
    date="2025-06-12",
    ag_key="TABM",
    model_key="TABM",
    config_default="TabM_GPU_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix="_GPU",
    verified=True,
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
