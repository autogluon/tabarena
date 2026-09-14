from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabiclm.hpo import gen_tabiclm
from tabarena.models.tabiclm.model import TabICLMModel

tabiclm_descriptor = ModelDescriptor(
    display_name="TabICL-M",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/Sompote/TabICL-M",
    date_introduced="2026-09-08",
)

tabiclm_method_metadata = tabiclm_descriptor.method_metadata(
    method="TabICL-M",
    date="2026-09-08",
    ag_key="TA-TABICLM",
    model_key="TABICLM",
    config_default="TabICLM_c1_default_BAG_L1",
    can_hpo=False,
    suite="tabarena-2026-09-08",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

tabiclm_info = ModelInfo(
    model_cls=TabICLMModel,
    search_space=gen_tabiclm,
    method_metadata=tabiclm_method_metadata,
    pip_extra=("tabicl-m @ git+https://github.com/Sompote/TabICL-M.git@65b617714f348acdd391485be65ccf6fb8b2fece",),
    prefetch_weights=TabICLMModel.prefetch_weights,
)
