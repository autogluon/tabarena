from __future__ import annotations

from tabarena.models.mitra_v2.model import MitraV2Model
from tabarena.utils.config_utils import ConfigGenerator

# The recipe is frozen (see `_internal/recipe.py`): the default configuration is the method.
gen_mitra_v2 = ConfigGenerator(
    model_cls=MitraV2Model,
    manual_configs=[{}],
    search_space={},
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_mitra_v2.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
