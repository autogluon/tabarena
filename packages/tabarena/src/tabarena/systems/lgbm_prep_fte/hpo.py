from __future__ import annotations

from tabarena.systems.lgbm_prep_fte.system import NAME, LGBMPrepFTESystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# One fixed portfolio per benchmark, so one config each: the bags follow the validation and preprocessing
# TabArena applies to models there. The budget is the run's `time_limit`; `system.system_time_limit(arena)` is
# the one under which every bag gets a model config's budget.
gen_lgbm_prep_fte = SystemConfigGenerator(
    model_cls=LGBMPrepFTESystemModel,
    name=NAME,
    manual_configs=[{"arena": "tabarena"}],
)
gen_lgbm_prep_fte_beyondarena = SystemConfigGenerator(
    model_cls=LGBMPrepFTESystemModel,
    name=NAME,
    manual_configs=[{"arena": "beyondarena"}],
)
