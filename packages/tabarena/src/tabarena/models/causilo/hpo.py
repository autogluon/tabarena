from __future__ import annotations

from tabarena.models.causilo.model import CausiloModel
from tabarena.utils.config_utils import ConfigGenerator

# One fixed release recipe; explicit seed remains overridable through the wrapper.
gen_causilo = ConfigGenerator(
    model_cls=CausiloModel,
    manual_configs=[{"random_state": 42}],
    search_space={},
)
