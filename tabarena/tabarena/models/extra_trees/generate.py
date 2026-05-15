"""Back-compat shim: `gen_extratrees` now lives in `tabarena.models.extra_trees.hpo`."""

from __future__ import annotations

from tabarena.models.extra_trees.hpo import gen_extratrees, generate_configs_xt

__all__ = ["gen_extratrees", "generate_configs_xt"]
