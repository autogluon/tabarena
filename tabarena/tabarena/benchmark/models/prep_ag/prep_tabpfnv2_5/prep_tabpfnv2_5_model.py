from __future__ import annotations

from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.benchmark.models.prep_ag.prep_mixin import ModelAgnosticPrepMixin


class PrepRealTabPFNv25Model(ModelAgnosticPrepMixin, RealTabPFNv25Model):
    ag_key = "prep_REALTABPFN-V2.5"
    ag_name = "prep_RealTabPFN-v2.5"
