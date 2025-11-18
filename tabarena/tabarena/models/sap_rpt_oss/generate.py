from __future__ import annotations

from tabarena.benchmark.models.ag.sap_rpt_oss.sap_rpt_oss_model import SAPRPTOSSModel
from tabarena.utils.config_utils import ConfigGenerator

gen_sap_rpt_oss = ConfigGenerator(
    model_cls=SAPRPTOSSModel,
    manual_configs=[{}],
    search_space={},  # No search space
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_sap_rpt_oss.generate_all_bag_experiments(
                num_random_configs=0
            ),
        ),
    )
