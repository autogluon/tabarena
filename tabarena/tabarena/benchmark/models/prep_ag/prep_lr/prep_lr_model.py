from __future__ import annotations

from autogluon.tabular.models.lr.lr_model import LinearModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin

class PrepLinearModel(ModelAgnosticPrepMixin, LinearModel):
    ag_key = "prep_LR"
    ag_name = "prep_LinearModel"
