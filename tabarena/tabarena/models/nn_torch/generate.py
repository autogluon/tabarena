"""Back-compat shim: `gen_nn_torch` now lives in `tabarena.models.nn_torch.hpo`."""

from __future__ import annotations

from tabarena.models.nn_torch.hpo import gen_nn_torch, generate_configs_nn_torch

__all__ = ["gen_nn_torch", "generate_configs_nn_torch"]
