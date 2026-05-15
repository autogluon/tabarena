"""Back-compat shim: `gen_randomforest` now lives in `tabarena.models.random_forest.hpo`."""

from __future__ import annotations

from tabarena.models.random_forest.hpo import gen_randomforest, generate_configs_rf

__all__ = ["gen_randomforest", "generate_configs_rf"]
