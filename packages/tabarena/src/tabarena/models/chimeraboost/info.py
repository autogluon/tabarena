from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.chimeraboost.hpo import gen_chimeraboost
from tabarena.models.chimeraboost.model import ChimeraBoostModel

chimeraboost_method_metadata = MethodMetadata(
    method="ChimeraBoost",
    display_name="ChimeraBoost",
    method_type="config",
    compute="cpu",
    date="2026-06-15",
    ag_key="CHIMERA",
    config_default="ChimeraBoost_c1_BAG_L1",
    can_hpo=True,
    is_bag=False,
    verified=False,
    reference_url="https://github.com/bbstats/chimeraboost",
    # has_raw/has_processed/has_results + s3_bucket/s3_prefix/cache_type are set by
    # the maintainers when the result artifacts are hosted in the official pool.
)

chimeraboost_info = ModelInfo(
    model_cls=ChimeraBoostModel,
    search_space=gen_chimeraboost,
    method_metadata=chimeraboost_method_metadata,
    pip_extra=("chimeraboost>=0.13.0",),
)
