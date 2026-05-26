"""Shim for the relocated TabICL model wrappers.

The implementation now lives at `tabarena.models.tabicl.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabicl.tabicl_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabicl.model import TabICLModel, TabICLv2Model

__all__ = ["TabICLModel", "TabICLv2Model"]
