from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.chimeraboost.hpo import gen_chimeraboost
from tabarena.models.chimeraboost.model import ChimeraBoostModel

chimeraboost_method_metadata = MethodMetadata.config(
    method="ChimeraBoost",
    display_name="ChimeraBoost",
    compute="cpu",
    date="2026-06-15",
    ag_key="CHIMERA",
    config_default="ChimeraBoost_c1_BAG_L1",
    is_bag=False,
    verified=False,
    reference_url="https://github.com/bbstats/chimeraboost",
    # Not yet hosted: bucket/prefix/cache_type are set by the maintainers once the result
    # artifacts land in the official pool; until then cache_type infers "local" (no remote
    # location set). has_raw/has_processed/has_results stay True — a config has all three
    # artifact tiers, hosted or not.
)

chimeraboost_info = ModelInfo(
    model_cls=ChimeraBoostModel,
    search_space=gen_chimeraboost,
    method_metadata=chimeraboost_method_metadata,
    pip_extra=("chimeraboost>=0.13.0",),
)
