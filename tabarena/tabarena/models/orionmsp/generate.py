"""Back-compat shim: `gen_orionmsp` now lives in `tabarena.models.orionmsp.hpo`."""

from __future__ import annotations

from tabarena.models.orionmsp.hpo import gen_orionmsp

__all__ = ["gen_orionmsp"]
