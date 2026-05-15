"""Back-compat shim: `gen_limix` now lives in `tabarena.models.limix.hpo`."""

from __future__ import annotations

from tabarena.models.limix.hpo import gen_limix

__all__ = ["gen_limix"]
