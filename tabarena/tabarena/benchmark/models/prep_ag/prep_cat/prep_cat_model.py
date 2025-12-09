from __future__ import annotations

from autogluon.tabular.models.catboost.catboost_model import CatBoostModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepCatBoostModel(ModelAgnosticPrepMixin, CatBoostModel):
    ag_key = "prep_CAT"
    ag_name = "prep_CatBoost"
