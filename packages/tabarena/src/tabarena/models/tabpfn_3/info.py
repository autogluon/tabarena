from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfn_3.hpo import gen_tabpfn_3
from tabarena.models.tabpfn_3.model import TabPFN3Model, prefetch_weights

tabpfn_3_descriptor = ModelDescriptor(
    display_name="TabPFN-3",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2605.13986",
)

tabpfn_3_method_metadata = tabpfn_3_descriptor.method_metadata(
    method="TabPFN-3",
    ag_key="TA-TABPFN-3",
    config_default="TabPFN-3_c1_BAG_L1",
    can_hpo=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    date="2026-05-13",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    s3_bucket="tabarena",
    s3_prefix="cache",
    verified=True,
)


tabpfn_3_info = ModelInfo(
    model_cls=TabPFN3Model,
    search_space=gen_tabpfn_3,
    method_metadata=tabpfn_3_method_metadata,
    pip_extra=("tabpfn>=8.0.8",),
    prefetch_weights=prefetch_weights,
)
