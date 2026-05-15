"""Back-compat shim: `gen_linear` now lives in `tabarena.models.lr.hpo`."""

from __future__ import annotations

from tabarena.models.lr.hpo import gen_linear, generate_configs_lr

__all__ = ["gen_linear", "generate_configs_lr"]
