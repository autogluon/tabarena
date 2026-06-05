from __future__ import annotations

from autogluon.tabular.models import XGBoostModel

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.xgboost.hpo import gen_xgboost

xgboost_method_metadata = MethodMetadata(
    method="XGBoost",
    method_type="config",
    display_name="XGBoost",
    compute="cpu",
    date="2025-06-12",
    date_introduced="2014-03",
    ag_key="XGB",
    config_default="XGBoost_c1_BAG_L1",
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
    reference_url="https://arxiv.org/abs/1603.02754",
)


xgboost_info = ModelInfo(
    model_cls=XGBoostModel,
    search_space=gen_xgboost,
    method_metadata=xgboost_method_metadata,
)
