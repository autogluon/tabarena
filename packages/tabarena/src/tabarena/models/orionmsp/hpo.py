from __future__ import annotations

from tabarena.models.orionmsp.model import OrionMSPModel
from tabarena.utils.config_utils import ConfigGenerator

gen_orionmsp = ConfigGenerator(
    model_cls=OrionMSPModel,
    manual_configs=[{}],
    search_space={},
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_orionmsp.generate_all_bag_experiments(
                num_random_configs=0,
            ),
        ),
    )
