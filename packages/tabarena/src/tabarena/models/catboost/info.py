from __future__ import annotations

from autogluon.tabular.models import CatBoostModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.catboost.hpo import gen_catboost

catboost_method_metadata = MethodMetadata(
    method="CatBoost",
    method_type="config",
    display_name="CatBoost",
    compute="cpu",
    date="2025-06-12",
    date_introduced="2017-06",
    ag_key="CAT",
    config_default="CatBoost_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    name_suffix=None,
    # FIXME: technically GBDTs are not verified
    verified=True,
    reference_url="https://arxiv.org/abs/1706.09516",
)


catboost_info = ModelInfo(
    model_cls=CatBoostModel,
    search_space=gen_catboost,
    method_metadata=catboost_method_metadata,
)
