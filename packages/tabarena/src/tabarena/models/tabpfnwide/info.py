from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfnwide.hpo import gen_tabpfnwide
from tabarena.models.tabpfnwide.model import TabPFNWideModel

tabpfnwide_method_metadata = MethodMetadata.config(
    method="TabPFN-Wide",
    display_name="TabPFN-Wide",
    compute="gpu",
    date="2026-05-13",
    ag_key="TA-TABPFN-WIDE",
    config_default="TabPFN-Wide_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    verified=False,
    reference_url="https://arxiv.org/abs/2510.06162",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    s3_bucket="tabarena",
    s3_prefix="cache",
)


tabpfnwide_info = ModelInfo(
    model_cls=TabPFNWideModel,
    search_space=gen_tabpfnwide,
    method_metadata=tabpfnwide_method_metadata,
    pip_extra=("tabpfnwide>=0.3.0",),
)
