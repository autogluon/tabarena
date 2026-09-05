from __future__ import annotations

from autogluon.common.space import Int, Real

from tabarena.models.aplr.model import APLRDeepIntModel, APLRTwoWayIntModel
from tabarena.utils.config_utils import ConfigGenerator

gen_aplr_two_way_int = ConfigGenerator(
    model_cls=APLRTwoWayIntModel,
    manual_configs=[{}],
    search_space={
        "penalty_for_interactions": Real(0, 1),
        "min_observations_in_split": Real(0.2, 0.6),
        "ridge_penalty": Real(0, 0.01),
    },
)


gen_aplr_deep_int = ConfigGenerator(
    model_cls=APLRDeepIntModel,
    manual_configs=[{}],
    search_space={
        "penalty_for_interactions": Real(0, 1),
        "min_observations_in_split": Real(0.2, 0.6),
        "ridge_penalty": Real(0, 0.01),
        "max_interaction_level": Int(1, 11),
    },
)
