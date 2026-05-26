"""Shim for the relocated LimiX model wrapper.

The implementation now lives at `tabarena.models.limix.model`. This module
re-exports it so existing imports of
`tabarena.benchmark.models.ag.limix.limix_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.limix.model import LimiXModel

__all__ = ["LimiXModel"]
