from __future__ import annotations

from autogluon.tabular.models import CatBoostModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.catboost.hpo import gen_catboost

catboost_descriptor = ModelDescriptor(
    display_name="CatBoost",
    compute="cpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/1706.09516",
)

catboost_method_metadata = catboost_descriptor.method_metadata(
    method="CatBoost",
    method_type="config",
    date="2025-06-12",
    ag_key="CAT",
    config_default="CatBoost_c1_BAG_L1",
    can_hpo=True,
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
)


catboost_info = ModelInfo(
    model_cls=CatBoostModel,
    search_space=gen_catboost,
    method_metadata=catboost_method_metadata,
)
