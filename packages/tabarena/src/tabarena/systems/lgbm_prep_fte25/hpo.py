from __future__ import annotations

from tabarena.systems.lgbm_prep_fte25.system import NAME, LGBMPrepFTE25SystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

# One fixed portfolio, so one config per benchmark: the bags follow the validation and preprocessing TabArena
# applies to models there. The time budget comes from the run's resources.
gen_lgbm_prep_fte25 = SystemConfigGenerator(
    model_cls=LGBMPrepFTE25SystemModel,
    name=NAME,
    manual_configs=[{"arena": "tabarena"}],
)
gen_lgbm_prep_fte25_beyondarena = SystemConfigGenerator(
    model_cls=LGBMPrepFTE25SystemModel,
    name=NAME,
    manual_configs=[{"arena": "beyondarena"}],
)
