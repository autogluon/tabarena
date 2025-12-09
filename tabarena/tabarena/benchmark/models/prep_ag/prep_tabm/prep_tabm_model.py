from __future__ import annotations

from autogluon.tabular.models.tabm.tabm_model import TabMModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepTabMModel(ModelAgnosticPrepMixin, TabMModel):
    ag_key = "prep_TABM"
    ag_name = "prep_TabM"
