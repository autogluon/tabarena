"""Shim for the relocated OrionMSP model wrapper.

The implementation now lives at `tabarena.models.orionmsp.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.orionmsp.orionmsp_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.orionmsp.model import OrionMSPModel

__all__ = ["OrionMSPModel"]
