"""Back-compat shim: `gen_tabdpt` now lives in `tabarena.models.tabdpt.hpo`."""

from __future__ import annotations

from tabarena.models.tabdpt.hpo import gen_tabdpt

__all__ = ["gen_tabdpt"]
