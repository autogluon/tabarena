from __future__ import annotations

from tabarena.benchmark.models.ag.ebm.ebm_model import ExplainableBoostingMachineModel
from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.benchmark.models.ag.tabdpt.tabdpt_model import TabDPTModel
from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLModel
from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.benchmark.models.ag.tabpfnv2.tabpfnv2_client_model import TabPFNV2ClientModel
from tabarena.benchmark.models.ag.tabpfnv2.tabpfnv2_model import TabPFNV2Model
from tabarena.benchmark.models.ag.xrfm.xrfm_model import XRFMModel
from tabarena.benchmark.models.ag.knn_new.knn_model import KNNNewModel
__all__ = [
    "ExplainableBoostingMachineModel",
    "ModernNCAModel",
    "RealMLPModel",
    "TabDPTModel",
    "TabICLModel",
    "TabMModel",
    "TabPFNV2ClientModel",
    "TabPFNV2Model",
    "XRFMModel",
    "KNNNewModel",
]
