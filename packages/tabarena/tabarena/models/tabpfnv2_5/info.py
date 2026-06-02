from __future__ import annotations

from tabarena.models.tabpfnv2_5.model import (
    RealTabPFNv25Model,
    TabPFNv26Model,
    prefetch_weights,
)
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfnv2_5.hpo import gen_realtabpfnv25, gen_tabpfnv26
from tabarena.models._method_metadata import MethodMetadata


realtabpfnv25_method_metadata = MethodMetadata(
    method="RealTabPFN-v2.5",
    method_type="config",
    display_name="RealTabPFN-2.5",
    compute="gpu",
    date="2025-11-12",
    ag_key="REALTABPFN-V2.5",
    model_key="REALTABPFN-V2.5",
    config_default="RealTabPFN-v2.5_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2025-11-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2511.08667",
)


tabpfnv26_method_metadata = MethodMetadata(
    method="TabPFN-v2.6",
    method_type="config",
    display_name="TabPFN-2.6",
    compute="gpu",
    date="2026-03-25",
    ag_key="TABPFN-V2.6",
    model_key="TABPFN-V2.6",
    config_default="TabPFN-v2.6_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-03-18",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2511.08667",
    cache_type="r2",
)


realtabpfnv25_info = ModelInfo(
    model_cls=RealTabPFNv25Model,
    search_space=gen_realtabpfnv25,
    method_metadata=realtabpfnv25_method_metadata,
    pip_extra=("tabpfn>=8.0.0",),
    prefetch_weights=prefetch_weights,
)


tabpfnv26_info = ModelInfo(
    model_cls=TabPFNv26Model,
    search_space=gen_tabpfnv26,
    method_metadata=tabpfnv26_method_metadata,
    pip_extra=("tabpfn>=8.0.0",),
    prefetch_weights=prefetch_weights,
)
