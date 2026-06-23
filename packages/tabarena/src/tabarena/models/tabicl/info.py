from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabicl.hpo import gen_tabicl, gen_tabiclv2
from tabarena.models.tabicl.model import (
    TabICLModel,
    TabICLv2Model,
)

tabicl_descriptor = ModelDescriptor(
    display_name="TabICL",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2502.05564",
)

tabiclv2_descriptor = ModelDescriptor(
    display_name="TabICLv2",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2602.11139",
)

tabicl_method_metadata = tabicl_descriptor.method_metadata(
    method="TabICL_GPU",
    date="2025-06-12",
    ag_key="TABICL",
    model_key="TABICL",
    config_default="TabICL_GPU_c1_BAG_L1",
    can_hpo=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix="_GPU",
    verified=True,
)


tabiclv2_method_metadata = tabiclv2_descriptor.method_metadata(
    method="TabICLv2",
    date="2026-02-16",
    ag_key="TABICLV2",
    model_key="TABICLV2",
    config_default="TabICLv2_c1_BAG_L1",
    can_hpo=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-02-16",
    cache_type="s3",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
)


tabicl_info = ModelInfo(
    model_cls=TabICLModel,
    search_space=gen_tabicl,
    method_metadata=tabicl_method_metadata,
    pip_extra=("tabicl>=2.0.0",),
    prefetch_weights=TabICLModel.prefetch_weights,
)


tabiclv2_info = ModelInfo(
    model_cls=TabICLv2Model,
    search_space=gen_tabiclv2,
    method_metadata=tabiclv2_method_metadata,
    pip_extra=("tabicl>=2.0.0",),
    prefetch_weights=TabICLv2Model.prefetch_weights,
)
