"""Shim for the relocated RealTabPFNv2.5 / TabPFNv2.6 model wrappers.

The implementation now lives at `tabarena.models.tabpfnv2_5.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabpfnv2_5.model import RealTabPFNv25Model, TabPFNv26Model

__all__ = ["RealTabPFNv25Model", "TabPFNv26Model"]
