from __future__ import annotations

from tabarena.models.tabpfn_3.model import TabPFN3Model
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfn_3.hpo import gen_tabpfn_3
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabpfn_3_method_metadata = MethodMetadata(
    method="TabPFN-3",
    method_type="config",
    display_name="TabPFN-3",
    compute="gpu",
    ag_key="TA-TABPFN-3",
    config_default="TabPFN-3_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    date="2026-05-13",
    reference_url="https://arxiv.org/abs/2605.13986",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    verified=True,
)


tabpfn_3_info = ModelInfo(
    model_cls=TabPFN3Model,
    search_space=gen_tabpfn_3,
    method_metadata=tabpfn_3_method_metadata,
    pip_extra=("tabpfn>=8.0.0",),
)
