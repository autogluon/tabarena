"""Back-compat shim: `gen_mitra` now lives in `tabarena.models.mitra.hpo`."""

from __future__ import annotations

from tabarena.models.mitra.hpo import gen_mitra

__all__ = ["gen_mitra"]
