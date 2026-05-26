"""Shim for the relocated TabPFN-Wide model wrapper.

The implementation now lives at `tabarena.models.tabpfnwide.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabpfnwide.tabpfnwide_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabpfnwide.model import TabPFNWideModel

__all__ = ["TabPFNWideModel"]
