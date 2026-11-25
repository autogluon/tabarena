from __future__ import annotations

from tabarena.models.causilo.model import Causilo15Model, CausiloModel
from tabarena.utils.config_utils import ConfigGenerator

# One fixed release recipe per version. Causilo pins seed 42 for every fold and the refit model, as its
# hosted runs did, so a rerun reproduces them; a ``random_state`` here overrides the bag's per-fold seeds.
gen_causilo = ConfigGenerator(
    model_cls=CausiloModel,
    manual_configs=[{"random_state": 42}],
    search_space={},
)

# Causilo-1.5 takes its seed from the bag through ``seed_name``, so the fold models vary like every entrant's.
gen_causilo15 = ConfigGenerator(
    model_cls=Causilo15Model,
    manual_configs=[{}],
    search_space={},
)
