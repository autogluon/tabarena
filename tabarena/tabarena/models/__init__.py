from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.models._model_info import ModelInfo
from tabarena.models._registry import (
    discover_models,
    get_model_registry,
    register_model_info,
)

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata
    from tabarena.models.iltm.model import ILTMModel
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


# Maps top-level public name -> module to import it from. Resolved lazily by
# `__getattr__` so that `import tabarena.models` stays cheap — model wrappers
# pull in heavy ML libraries, and `_method_metadata.py` carries S3 / paper /
# repository transitive imports we don't want to charge every consumer for.
_LAZY_CLASSES: dict[str, str] = {
    "MethodMetadata": "tabarena.models._method_metadata",
    "ILTMModel": "tabarena.models.iltm.model",
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


# Names imported eagerly at the top of this module. Kept as a tuple so
# `__all__` can be derived alongside the lazy entries with one source of
# truth per surface (eager vs lazy).
_EAGER_EXPORTS = (
    "ModelInfo",
    "discover_models",
    "get_model_registry",
    "register_model_info",
)

# `__all__` is derived: any name added to `_LAZY_CLASSES` or `_EAGER_EXPORTS`
# is automatically part of the public surface. The `TYPE_CHECKING` block
# above still needs static `from … import …` lines so IDEs and static
# analysers can resolve lazy names — that's the one duplication we can't
# fold away.
__all__ = sorted({*_LAZY_CLASSES, *_EAGER_EXPORTS})
