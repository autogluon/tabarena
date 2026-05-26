"""Shim for the relocated ModernNCA model wrapper.

The implementation now lives at `tabarena.models.modernnca.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.modernnca.modernnca_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.modernnca.model import ModernNCAModel

__all__ = ["ModernNCAModel"]
