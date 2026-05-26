from __future__ import annotations

from tabarena.models.ebm.model import ExplainableBoostingMachineModel
from tabarena.benchmark.models.ag.knn_new.knn_model import KNNNewModel
from tabarena.benchmark.models.ag.limix.limix_model import LimiXModel
from tabarena.benchmark.models.ag.modernnca.modernnca_model import ModernNCAModel
from tabarena.models.orionmsp.model import OrionMSPModel
from tabarena.models.perpetual_booster.model import (
    PerpetualBoosterModel,
)
from tabarena.models.realmlp.model import RealMLPModel
from tabarena.models.sap_rpt_oss.model import SAPRPTOSSModel
from tabarena.models.tabdpt.model import TabDPTModel
from tabarena.models.tabicl.model import TabICLModel, TabICLv2Model
from tabarena.benchmark.models.ag.tabm.tabm_model import TabMModel
from tabarena.models.tabpfnv2_5.model import (
    RealTabPFNv25Model,
    TabPFNv26Model,
)
from tabarena.models.tabpfn_3.model import TabPFN3Model
from tabarena.models.tabpfnwide.model import TabPFNWideModel
from tabarena.models.tabstar.model import TabSTARModel
from tabarena.models.xrfm.model import XRFMModel

__all__ = [
    "ExplainableBoostingMachineModel",
    "KNNNewModel",
    "LimiXModel",
    "ModernNCAModel",
    "OrionMSPModel",
    "PerpetualBoosterModel",
    "RealMLPModel",
    "RealTabPFNv25Model",
    "SAPRPTOSSModel",
    "TabDPTModel",
    "TabICLModel",
    "TabICLv2Model",
    "TabMModel",
    "TabPFN3Model",
    "TabPFNWideModel",
    "TabPFNv26Model",
    "TabSTARModel",
    "XRFMModel",
]
