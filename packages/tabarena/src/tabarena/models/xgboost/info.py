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
    date="2025-06-12",
    ag_key="XGB",
    config_default="XGBoost_c1_BAG_L1",
    can_hpo=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    artifact_name="tabarena-2025-06-12",
    cache_type="s3",
    bucket="tabarena",
    prefix="cache",
    cache_kwargs={"upload_as_public": True},
    name_suffix=None,
    # FIXME: technically GBDTs are not verified
    verified=True,
)


xgboost_info = ModelInfo(
    model_cls=XGBoostModel,
    search_space=gen_xgboost,
    method_metadata=xgboost_method_metadata,
)
