from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.limix.hpo import gen_limix
from tabarena.models.limix.model import LimiXModel

limix_method_metadata = MethodMetadata.config(
    method="LimiX",
    display_name="LimiX",
    compute="gpu",
    date="2026-05-13",
    ag_key="TA-LIMIX",
    config_default="LimiX_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=False,
    reference_url="https://arxiv.org/abs/2509.03505",
    cache_type="r2",
    artifact_name="tabarena-2026-05-13",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    has_raw=True,
    has_processed=True,
    has_results=True,
)


limix_info = ModelInfo(
    model_cls=LimiXModel,
    search_space=gen_limix,
    method_metadata=limix_method_metadata,
    pip_extra=("einops", "kditransform"),
    prefetch_weights=LimiXModel.prefetch_weights,
)
