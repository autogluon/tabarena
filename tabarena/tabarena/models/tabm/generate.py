"""Back-compat shim: `gen_tabm` now lives in `tabarena.models.tabm.hpo`."""

from __future__ import annotations

from tabarena.models.tabm.hpo import (
    gen_tabm,
    generate_configs_tabm,
    generate_single_config_tabm,
)

__all__ = ["gen_tabm", "generate_configs_tabm", "generate_single_config_tabm"]
