from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nori.hpo import gen_nori
from tabarena.models.nori.model import NoriModel

nori_method_metadata = MethodMetadata.config(
    method="Nori",
    display_name="Nori",
    compute="gpu",
    date="2026-06-18",
    ag_key="TA-NORI",
    config_default="Nori_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=False,
    reference_url="https://github.com/Synthefy/synthefy-nori",
    # has_raw/has_processed/has_results + s3_bucket/s3_prefix/cache_type are set by
    # the maintainers when the result artifacts are hosted in the official pool.
)

nori_info = ModelInfo(
    model_cls=NoriModel,
    search_space=gen_nori,
    method_metadata=nori_method_metadata,
    pip_extra=("synthefy-nori>=0.7.0",),
    prefetch_weights=NoriModel.prefetch_weights,
)
