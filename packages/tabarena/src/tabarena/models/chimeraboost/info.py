from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.chimeraboost.hpo import gen_chimeraboost
from tabarena.models.chimeraboost.model import ChimeraBoostModel, prefetch_weights

chimeraboost_method_metadata = MethodMetadata.config(
    method="ChimeraBoost",
    ag_key="CHIMERA",
    compute="cpu",
    is_bag=True,
    can_hpo=True,
    config_default="ChimeraBoost_c1_default_BAG_L1",
    suite="tabarena-2026-06-30",
    date="2026-06-15",
    reference_url="https://github.com/bbstats/chimeraboost",
    display_name="ChimeraBoost",
    verified=True,
    cache_type="r2",  # one of: "local", "r2", "s3"
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},  # only if uploading (s3 adds "upload_as_public": True)
)

chimeraboost_info = ModelInfo(
    model_cls=ChimeraBoostModel,
    search_space=gen_chimeraboost,
    method_metadata=chimeraboost_method_metadata,
    pip_extra=("chimeraboost>=0.14.1",),
    prefetch_weights=prefetch_weights,
)
