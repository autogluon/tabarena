from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nori.hpo import gen_nori
from tabarena.models.nori.model import NoriModel

nori_method_metadata = MethodMetadata.config(
    method="Nori",
    suite="tabarena-2026-06-18",
    ag_key="TA-NORI",
    config_default="Nori_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-06-18",
    reference_url="https://github.com/Synthefy/synthefy-nori",
    display_name="Nori",
    verified=False,
    # Not yet hosted: bucket/prefix/cache_type are set by the maintainers once the result
    # artifacts land in the official pool; until then cache_type infers "local" (no remote
    # location set). has_raw/has_processed/has_results stay True — a config has all three
    # artifact tiers, hosted or not.
)

nori_info = ModelInfo(
    model_cls=NoriModel,
    search_space=gen_nori,
    method_metadata=nori_method_metadata,
    pip_extra=("synthefy-nori>=0.7.0",),
    prefetch_weights=NoriModel.prefetch_weights,
)
