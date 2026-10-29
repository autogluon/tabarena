from __future__ import annotations

from tabarena.models.tababax.model import TababaxModel
from tabarena.utils.config_utils import ConfigGenerator

# One released checkpoint. The model class supplies the recommended defaults; this config does not
# override them. Pass model_path to use a local file instead of the Hugging Face download.
gen_tababax = ConfigGenerator(
    model_cls=TababaxModel,
    search_space={},
    manual_configs=[{}],
)
