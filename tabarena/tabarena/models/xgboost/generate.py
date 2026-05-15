"""Back-compat shim: `gen_xgboost` now lives in `tabarena.models.xgboost.hpo`."""

from __future__ import annotations

from tabarena.models.xgboost.hpo import gen_xgboost, generate_configs_xgboost

__all__ = ["gen_xgboost", "generate_configs_xgboost"]
