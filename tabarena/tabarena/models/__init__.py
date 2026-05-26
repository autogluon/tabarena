from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.models._model_info import ModelInfo
from tabarena.models._registry import (
    discover_models,
    get_model_registry,
    register_model_info,
)

if TYPE_CHECKING:
    from tabarena.models.ebm.model import ExplainableBoostingMachineModel
    from tabarena.models.knn.model import KNNNewModel
    from tabarena.models.limix.model import LimiXModel
    from tabarena.models.modernnca.model import ModernNCAModel
    from tabarena.models.orionmsp.model import OrionMSPModel
    from tabarena.models.perpetual_booster.model import PerpetualBoosterModel
    from tabarena.models.realmlp.model import RealMLPModel
    from tabarena.models.sap_rpt_oss.model import SAPRPTOSSModel
    from tabarena.models.tabdpt.model import TabDPTModel
    from tabarena.models.tabicl.model import TabICLModel, TabICLv2Model
    from tabarena.models.tabm.model import TabMModel
    from tabarena.models.tabpfn_3.model import TabPFN3Model
    from tabarena.models.tabpfnv2_5.model import RealTabPFNv25Model, TabPFNv26Model
    from tabarena.models.tabpfnwide.model import TabPFNWideModel
    from tabarena.models.tabstar.model import TabSTARModel
    from tabarena.models.xrfm.model import XRFMModel


# Maps top-level class name -> "tabarena.models.<key>.model" submodule.
# Resolved lazily by `__getattr__` so that `import tabarena.models` does not
# transitively load every per-model `info.py` (which would trip a latent
# circular import via `nips2025_utils.artifacts._tabarena_method_metadata`).
_LAZY_CLASSES: dict[str, str] = {
    "ExplainableBoostingMachineModel": "tabarena.models.ebm.model",
    "KNNNewModel": "tabarena.models.knn.model",
    "LimiXModel": "tabarena.models.limix.model",
    "ModernNCAModel": "tabarena.models.modernnca.model",
    "OrionMSPModel": "tabarena.models.orionmsp.model",
    "PerpetualBoosterModel": "tabarena.models.perpetual_booster.model",
    "RealMLPModel": "tabarena.models.realmlp.model",
    "RealTabPFNv25Model": "tabarena.models.tabpfnv2_5.model",
    "SAPRPTOSSModel": "tabarena.models.sap_rpt_oss.model",
    "TabDPTModel": "tabarena.models.tabdpt.model",
    "TabICLModel": "tabarena.models.tabicl.model",
    "TabICLv2Model": "tabarena.models.tabicl.model",
    "TabMModel": "tabarena.models.tabm.model",
    "TabPFN3Model": "tabarena.models.tabpfn_3.model",
    "TabPFNWideModel": "tabarena.models.tabpfnwide.model",
    "TabPFNv26Model": "tabarena.models.tabpfnv2_5.model",
    "TabSTARModel": "tabarena.models.tabstar.model",
    "XRFMModel": "tabarena.models.xrfm.model",
}


def __getattr__(name: str):
    module_path = _LAZY_CLASSES.get(name)
    if module_path is None:
        raise AttributeError(f"module 'tabarena.models' has no attribute {name!r}")
    import importlib

    obj = getattr(importlib.import_module(module_path), name)
    globals()[name] = obj  # cache so subsequent lookups skip __getattr__
    return obj


__all__ = [
    "ExplainableBoostingMachineModel",
    "KNNNewModel",
    "LimiXModel",
    "ModelInfo",
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
    "discover_models",
    "get_model_registry",
    "register_model_info",
]
