from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabstar.hpo import gen_tabstar
from tabarena.models.tabstar.model import TabSTARModel, prefetch_weights

tabstar_method_metadata = MethodMetadata.config(
    method="TabSTAR",
    display_name="TabSTAR",
    compute="gpu",
    date="2026-03-02",
    ag_key="TABSTAR",
    config_default="TabSTAR_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    artifact_name="tabarena-2026-03-18",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2505.18125",
    cache_type="r2",
)


tabstar_info = ModelInfo(
    model_cls=TabSTARModel,
    search_space=gen_tabstar,
    method_metadata=tabstar_method_metadata,
    pip_extra=("tabstar==1.1.15",),
    prefetch_weights=prefetch_weights,
)
