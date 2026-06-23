from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabdpt.hpo import gen_tabdpt
from tabarena.models.tabdpt.model import TabDPTModel, prefetch_weights

tabdpt_descriptor = ModelDescriptor(
    display_name="TabDPT",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2410.18164",
)

tabdpt_method_metadata = tabdpt_descriptor.method_metadata(
    method="TabDPT_GPU",
    date="2025-10-20",
    ag_key="TABDPT",
    model_key="TABDPT_GPU",
    config_default="TabDPT_GPU_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-10-20",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    cache_kwargs={"upload_as_public": True},
    has_results=True,
    name_suffix=None,
    verified=True,
)


tabdpt_info = ModelInfo(
    model_cls=TabDPTModel,
    search_space=gen_tabdpt,
    method_metadata=tabdpt_method_metadata,
    pip_extra=("tabdpt>=1.1.10",),
    prefetch_weights=prefetch_weights,
)
