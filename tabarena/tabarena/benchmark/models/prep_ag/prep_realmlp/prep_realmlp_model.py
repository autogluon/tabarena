from __future__ import annotations

# from autogluon.tabular.models.realmlp.realmlp_model import RealMLPModel
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepRealMLPModel(ModelAgnosticPrepMixin, RealMLPModel):
    ag_key = "prep_REALMLP"
    ag_name = "prep_RealMLP"