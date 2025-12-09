from __future__ import annotations

from autogluon.tabular.models.xgboost.xgboost_model import XGBoostModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepXGBoostModel(ModelAgnosticPrepMixin, XGBoostModel):
    ag_key = "prep_XGB"
    ag_name = "prep_XGBoost"
