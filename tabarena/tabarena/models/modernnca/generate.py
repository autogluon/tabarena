"""Back-compat shim: `gen_modernnca` now lives in `tabarena.models.modernnca.hpo`."""

from __future__ import annotations

from tabarena.models.modernnca.hpo import (
    gen_modernnca,
    generate_configs_modernnca,
    generate_single_config_modernnca,
)

__all__ = [
    "gen_modernnca",
    "generate_configs_modernnca",
    "generate_single_config_modernnca",
]
