from __future__ import annotations

from tabarena.models.kumo_cellridge.model import KumoCellRidgeModel
from tabarena.utils.config_utils import ConfigGenerator

gen_kumo_cellridge = ConfigGenerator(model_cls=KumoCellRidgeModel, search_space={}, manual_configs=[{}])
