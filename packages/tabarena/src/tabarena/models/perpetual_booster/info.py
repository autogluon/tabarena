from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.perpetual_booster.hpo import gen_perpetual_booster
from tabarena.models.perpetual_booster.model import (
    PerpetualBoosterModel,
)

perpetual_booster_method_metadata = MethodMetadata.config(
    method="PerpetualBooster",
    display_name="PerpetualBooster",
    compute="cpu",
    date="2026-03-06",
    ag_key="PB",
    config_default="PerpetualBooster_c1_BAG_L1",
    is_bag=True,
    artifact_name="tabarena-2026-03-18",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    reference_url="https://perpetual-ml.com/",
    cache_type="r2",
)


perpetual_booster_info = ModelInfo(
    model_cls=PerpetualBoosterModel,
    search_space=gen_perpetual_booster,
    method_metadata=perpetual_booster_method_metadata,
    pip_extra=("perpetual",),
)
