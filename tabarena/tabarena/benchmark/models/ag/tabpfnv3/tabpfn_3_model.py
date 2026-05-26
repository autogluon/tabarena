"""Shim for the relocated TabPFN-3 model wrapper.

The implementation now lives at `tabarena.models.tabpfn_3.model`. This
module re-exports it so existing imports of
`tabarena.benchmark.models.ag.tabpfnv3.tabpfn_3_model` continue to work.
"""

from __future__ import annotations

from tabarena.models.tabpfn_3.model import TabPFN3Model

__all__ = ["TabPFN3Model"]
