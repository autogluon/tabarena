from __future__ import annotations

from tabarena.models.seldon.model import SeldonMiniModel, SeldonModel
from tabarena.utils.config_utils import ConfigGenerator

#: Each model runs one config, the released defaults from neuralk-oss's packaged config.
gen_seldon = ConfigGenerator(model_cls=SeldonModel, search_space={}, manual_configs=[{}])
gen_seldon_mini = ConfigGenerator(model_cls=SeldonMiniModel, search_space={}, manual_configs=[{}])
