from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.prismboost.hpo import gen_prismboost
from tabarena.models.prismboost.model import PrismBoostModel

prismboost_method_metadata = MethodMetadata.config(
    method="PrismBoost",
    display_name="PrismBoost",
    ag_key="PRISMBOOST",
    model_key="PRISMBOOST",
    compute="cpu",
    is_bag=True,
    can_hpo=True,
    config_default="PrismBoost_c1_default_BAG_L1",
    validation_protocol="8x1",
    suite="tabarena-2026-09-19",
    date="2026-09-19",
    date_introduced="2026-09",
    reference_url="https://github.com/PrismBoost/PrismBoost",
    license="MIT",
    verified=False,
    cache_type="local",
)

prismboost_info = ModelInfo(
    model_cls=PrismBoostModel,
    search_space=gen_prismboost,
    method_metadata=prismboost_method_metadata,
    pip_extra=("prismboost>=0.5.0",),
)
