"""Back-compat shim: `gen_lightgbm` now lives in `tabarena.models.lightgbm.hpo`."""

from __future__ import annotations

from tabarena.models.lightgbm.hpo import gen_lightgbm, generate_configs_lightgbm

__all__ = ["gen_lightgbm", "generate_configs_lightgbm"]
