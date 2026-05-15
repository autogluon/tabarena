"""Back-compat shim: `gen_knn` now lives in `tabarena.models.knn.hpo`."""

from __future__ import annotations

from tabarena.models.knn.hpo import gen_knn, generate_configs_knn

__all__ = ["gen_knn", "generate_configs_knn"]
