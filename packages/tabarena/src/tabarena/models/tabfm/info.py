from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabfm.hpo import gen_tabfm
from tabarena.models.tabfm.model import TabFMModel, prefetch_weights

tabfm_method_metadata = MethodMetadata.config(
    method="TabFM",
    suite="tabarena-2026-06-26",
    ag_key="TA-TABFM",
    model_key="TABFM",
    config_default="TabFM_c1_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-06-26",
    reference_url="https://github.com/google-research/tabfm",
    display_name="TabFM",
    verified=False,
)


tabfm_info = ModelInfo(
    model_cls=TabFMModel,
    search_space=gen_tabfm,
    method_metadata=tabfm_method_metadata,
    pip_extra=(
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@53f3fcfb8a3355f55c9fb49f04fbb62b8ba29109",
    ),
    prefetch_weights=prefetch_weights,
)
