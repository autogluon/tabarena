from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfnwide.hpo import gen_tabpfnwide
from tabarena.models.tabpfnwide.model import TabPFNWideModel

tabpfnwide_method_metadata = MethodMetadata.config(
    method="TabPFN-Wide",
    suite="tabarena-2026-05-13",
    ag_key="TA-TABPFN-WIDE",
    config_default="TabPFN-Wide_c1_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-05-13",
    reference_url="https://arxiv.org/abs/2510.06162",
    display_name="TabPFN-Wide",
    verified=False,
)


tabpfnwide_info = ModelInfo(
    model_cls=TabPFNWideModel,
    search_space=gen_tabpfnwide,
    method_metadata=tabpfnwide_method_metadata,
    pip_extra=("tabpfnwide>=0.3.0",),
)
