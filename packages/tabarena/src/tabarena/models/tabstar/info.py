from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabstar.hpo import gen_tabstar
from tabarena.models.tabstar.model import TabSTARModel, prefetch_weights

tabstar_method_metadata = MethodMetadata.config(
    method="TabSTAR",
    suite="tabarena-2026-03-18",
    ag_key="TABSTAR",
    config_default="TabSTAR_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-03-02",
    reference_url="https://arxiv.org/abs/2505.18125",
    display_name="TabSTAR",
)


tabstar_info = ModelInfo(
    model_cls=TabSTARModel,
    search_space=gen_tabstar,
    method_metadata=tabstar_method_metadata,
    pip_extra=("tabstar==1.1.15",),
    prefetch_weights=prefetch_weights,
)
