"""Back-compat shim: `gen_tabicl` / `gen_tabiclv2` now live in `tabarena.models.tabicl.hpo`."""

from __future__ import annotations

from tabarena.models.tabicl.hpo import gen_tabicl, gen_tabiclv2

__all__ = ["gen_tabicl", "gen_tabiclv2"]
