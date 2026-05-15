"""Back-compat shim: `gen_realtabpfnv25` / `gen_tabpfnv26` now live in `tabarena.models.tabpfnv2_5.hpo`."""

from __future__ import annotations

from tabarena.models.tabpfnv2_5.hpo import gen_realtabpfnv25, gen_tabpfnv26

__all__ = ["gen_realtabpfnv25", "gen_tabpfnv26"]
