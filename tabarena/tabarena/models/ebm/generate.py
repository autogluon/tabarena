"""Back-compat shim: `gen_ebm` now lives in `tabarena.models.ebm.hpo`."""

from __future__ import annotations

from tabarena.models.ebm.hpo import gen_ebm, generate_configs_ebm

__all__ = ["gen_ebm", "generate_configs_ebm"]
