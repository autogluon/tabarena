from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nori.hpo import gen_nori
from tabarena.models.nori.model import NoriModel

nori_method_metadata = MethodMetadata.config(
    method="Nori",
    suite="tabarena-2026-06-30",
    ag_key="TA-NORI",
    config_default="Nori_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-06-18",
    date_introduced="2026-06-12",
    reference_url="https://github.com/Synthefy/synthefy-nori",
    display_name="Nori",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

nori_info = ModelInfo(
    model_cls=NoriModel,
    search_space=gen_nori,
    method_metadata=nori_method_metadata,
    pip_extra=("synthefy-nori>=0.7.0",),
    prefetch_weights=NoriModel.prefetch_weights,
)
