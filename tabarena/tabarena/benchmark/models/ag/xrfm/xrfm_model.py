"""Shim for the relocated xRFM model wrapper.

The implementation now lives at `tabarena.models.xrfm.model`. This module
re-exports it so existing imports of
`tabarena.benchmark.models.ag.xrfm.xrfm_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.xrfm.model import XRFMModel

__all__ = ["XRFMModel"]
