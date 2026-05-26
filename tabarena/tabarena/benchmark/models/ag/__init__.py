"""Back-compat re-export of the model classes under the legacy ag namespace.

The classes themselves now live under `tabarena.models.<key>.model`, and are
collected at the top-level `tabarena.models` namespace. This module re-exports
them so existing `from tabarena.benchmark.models.ag import <Model>` imports
continue to work.
"""

from __future__ import annotations

from tabarena.models import (
    ExplainableBoostingMachineModel,
    KNNNewModel,
    LimiXModel,
    ModernNCAModel,
    OrionMSPModel,
    PerpetualBoosterModel,
    RealMLPModel,
    RealTabPFNv25Model,
    SAPRPTOSSModel,
    TabDPTModel,
    TabICLModel,
    TabICLv2Model,
    TabMModel,
    TabPFN3Model,
    TabPFNv26Model,
    TabPFNWideModel,
    TabSTARModel,
    XRFMModel,
)

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
