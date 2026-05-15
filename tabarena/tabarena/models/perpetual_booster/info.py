from __future__ import annotations

from tabarena.benchmark.models.ag.perpetual_booster.perpetual_booster_model import (
    PerpetualBoosterModel,
)
from tabarena.models._model_info import ModelInfo
from tabarena.models.perpetual_booster.hpo import gen_perpetual_booster
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


perpetual_booster_method_metadata = MethodMetadata(
    method="PerpetualBooster",
    method_type="config",
    display_name="PerpetualBooster",
    compute="cpu",
    date="2026-03-06",
    ag_key="PB",
    model_key="PB",
    config_default="PerpetualBooster_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-03-18",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://perpetual-ml.com/",
    cache_type="r2",
)


perpetual_booster_info = ModelInfo(
    model_cls=PerpetualBoosterModel,
    search_space=gen_perpetual_booster,
    method_metadata=perpetual_booster_method_metadata,
)
