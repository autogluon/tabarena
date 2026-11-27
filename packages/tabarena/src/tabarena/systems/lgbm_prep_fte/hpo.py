from __future__ import annotations

from tabarena.systems.lgbm_prep_fte.system import NAMES, LGBMPrepFTESystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# One fixed portfolio per benchmark, so one config and one leaderboard entry each: the bags follow the validation
# and preprocessing TabArena applies to models there. The budget is the run's `time_limit`;
# `system.system_time_limit(arena)` is the one under which every bag gets a model config's budget.
gen_lgbm_prep_fte200 = SystemConfigGenerator(
    model_cls=LGBMPrepFTESystemModel,
    name=NAMES["tabarena"],
    manual_configs=[{"arena": "tabarena"}],
)
gen_lgbm_prep_fte25 = SystemConfigGenerator(
    model_cls=LGBMPrepFTESystemModel,
    name=NAMES["beyondarena"],
    manual_configs=[{"arena": "beyondarena"}],
)
