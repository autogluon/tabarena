from __future__ import annotations

from autogluon.tabular.models import XGBoostModel

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.xgboost.hpo import gen_xgboost

xgboost_descriptor = ModelDescriptor(
    display_name="XGBoost",
    compute="cpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/1603.02754",
)

xgboost_method_metadata = xgboost_descriptor.method_metadata(
    method="XGBoost",
    suite="tabarena-2025-06-12",
    ag_key="XGB",
    config_default="XGBoost_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-06-12",
    # FIXME: technically GBDTs are not verified
)


xgboost_info = ModelInfo(
    model_cls=XGBoostModel,
    search_space=gen_xgboost,
    method_metadata=xgboost_method_metadata,
)
