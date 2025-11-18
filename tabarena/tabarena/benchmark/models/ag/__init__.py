from __future__ import annotations

from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from tabarena.benchmark.models.ag.knn_new.knn_model import KNNNewModel
from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import RealTabPFNv25Model
from tabarena.benchmark.models.ag.xrfm.xrfm_model import XRFMModel

__all__ = [
    "ExplainableBoostingMachineModel",
    "KNNNewModel",
    "ModernNCAModel",
    "RealMLPModel",
    "RealTabPFNv25Model",
    "TabDPTModel",
    "TabICLModel",
    "TabMModel",
    "XRFMModel",
]
