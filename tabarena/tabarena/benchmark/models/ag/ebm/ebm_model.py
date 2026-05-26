"""Shim for the relocated EBM model wrapper.

The implementation now lives at `tabarena.models.ebm.model`. This module
re-exports it so existing imports of
`tabarena.benchmark.models.ag.ebm.ebm_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.ebm.model import ExplainableBoostingMachineModel

__all__ = ["ExplainableBoostingMachineModel"]
