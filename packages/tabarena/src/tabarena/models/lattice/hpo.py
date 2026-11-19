from __future__ import annotations

from tabarena.models.lattice.model import LatticeModel, LatticeTinyModel
from tabarena.utils.config_utils import ConfigGenerator


def default_config() -> dict:
    """Cache the training context for the refit model only."""
    return {"kv_cache": False, "ag.refit_hyperparameters": {"kv_cache": True}}


gen_lattice = ConfigGenerator(model_cls=LatticeModel, search_space={}, manual_configs=[default_config()])
gen_lattice_tiny = ConfigGenerator(model_cls=LatticeTinyModel, search_space={}, manual_configs=[default_config()])
