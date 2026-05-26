"""Shim for the relocated RealMLP model wrapper.

The implementation now lives at `tabarena.models.realmlp.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.realmlp.realmlp_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.realmlp.model import RealMLPModel

__all__ = ["RealMLPModel"]
