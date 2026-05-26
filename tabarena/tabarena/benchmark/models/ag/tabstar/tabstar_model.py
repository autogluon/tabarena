"""Shim for the relocated TabSTAR model wrapper.

The implementation now lives at `tabarena.models.tabstar.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabstar.tabstar_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabstar.model import TabSTARModel

__all__ = ["TabSTARModel"]
