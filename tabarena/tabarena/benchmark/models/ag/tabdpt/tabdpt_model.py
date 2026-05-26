"""Shim for the relocated TabDPT model wrapper.

The implementation now lives at `tabarena.models.tabdpt.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabdpt.tabdpt_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabdpt.model import TabDPTModel

__all__ = ["TabDPTModel"]
