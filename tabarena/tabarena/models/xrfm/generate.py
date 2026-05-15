"""Back-compat shim: `gen_xrfm` now lives in `tabarena.models.xrfm.hpo`."""

from __future__ import annotations

from tabarena.models.xrfm.hpo import gen_xrfm, generate_configs_xrfm

__all__ = ["gen_xrfm", "generate_configs_xrfm"]
