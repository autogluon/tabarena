from __future__ import annotations

from tabarena.models.tabiclm.model import TabICLMModel
from tabarena.utils.config_utils import ConfigGenerator

# Default configuration only: TabICL-M is a zero-shot foundation model and is compared at its
# default like TabICLv2 (search over the shared preprocessing knobs is left to the TabICLv2 space).
gen_tabiclm = ConfigGenerator(model_cls=TabICLMModel, manual_configs=[{}], search_space={})
