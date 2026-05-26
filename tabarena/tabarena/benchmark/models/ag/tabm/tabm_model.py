"""Shim for the relocated TabM model wrapper.

The implementation now lives at `tabarena.models.tabm.model`. This module
re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabm.tabm_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabm.model import TabMModel

__all__ = ["TabMModel"]
