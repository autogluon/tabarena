from __future__ import annotations

from tabarena.models.tababax.model import TabAbaxModel
from tabarena.utils.config_utils import ConfigGenerator

# One released checkpoint, chosen by model_path at fit time. No search space.
gen_tababax = ConfigGenerator(
    model_cls=TabAbaxModel,
    search_space={},
    manual_configs=[{}],
)
