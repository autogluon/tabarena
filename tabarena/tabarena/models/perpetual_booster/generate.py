from __future__ import annotations

from autogluon.common.space import Categorical

from tabarena.benchmark.models.ag.perpetual_booster.perpetual_booster_model import (
    PerpetualBoosterModel,
)
from tabarena.utils.config_utils import ConfigGenerator

# Suggested search space from: https://github.com/perpetual-ml/perpetual/issues/66#issuecomment-3073175292
search_space = {"budget": Categorical(0.1, 0.2, 1.0, 1.5, 2.0)}

gen_perpetual_booster = ConfigGenerator(
    model_cls=PerpetualBoosterModel,
    search_space=search_space,
    manual_configs=[{"budget": 0.5}],
)

if __name__ == "__main__":
    print(gen_perpetual_booster.get_searcher_configs(5))
