from __future__ import annotations

from tabarena.systems.tabra_blends.system import TabraBlendsSystemModel
from tabarena.utils.config_utils import SystemConfigGenerator

gen_tabra_blends = SystemConfigGenerator(
    model_cls=TabraBlendsSystemModel,
    name="Tabra Blends",
    manual_configs=[{}],
)
