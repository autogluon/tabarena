"""Back-compat shim: `gen_realmlp` now lives in `tabarena.models.realmlp.hpo`."""

from __future__ import annotations

from tabarena.models.realmlp.hpo import (
    gen_realmlp,
    generate_configs_realmlp,
    generate_single_config_realmlp,
)

__all__ = ["gen_realmlp", "generate_configs_realmlp", "generate_single_config_realmlp"]
