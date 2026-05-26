"""Shim for the relocated KNNNew model wrapper.

The implementation now lives at `tabarena.models.knn.model`. This module
re-exports it so existing imports of
`tabarena.benchmark.models.ag.knn_new.knn_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.knn.model import KNNNewModel

__all__ = ["KNNNewModel"]
