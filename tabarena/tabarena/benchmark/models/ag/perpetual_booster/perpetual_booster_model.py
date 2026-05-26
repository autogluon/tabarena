"""Shim for the relocated PerpetualBooster model wrapper.

The implementation now lives at `tabarena.models.perpetual_booster.model`.
This module re-exports it so existing imports of
`tabarena.benchmark.models.ag.perpetual_booster.perpetual_booster_model`
continue to work.
"""

from __future__ import annotations

from tabarena.models.perpetual_booster.model import PerpetualBoosterModel

__all__ = ["PerpetualBoosterModel"]
