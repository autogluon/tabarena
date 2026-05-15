"""Back-compat shim: `gen_catboost` now lives in `tabarena.models.catboost.hpo`."""

from __future__ import annotations

from tabarena.models.catboost.hpo import gen_catboost, generate_configs_catboost

__all__ = ["gen_catboost", "generate_configs_catboost"]
