from __future__ import annotations

from tabarena.models.limix.model import LimiXModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.limix.hpo import gen_limix
from tabarena.models._method_metadata import MethodMetadata


limix_method_metadata = MethodMetadata(
    method="LimiX",
    method_type="config",
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
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
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
