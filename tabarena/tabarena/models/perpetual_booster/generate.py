"""Back-compat shim: `gen_perpetual_booster` now lives in `tabarena.models.perpetual_booster.hpo`."""

from __future__ import annotations

from tabarena.models.perpetual_booster.hpo import gen_perpetual_booster

__all__ = ["gen_perpetual_booster"]
