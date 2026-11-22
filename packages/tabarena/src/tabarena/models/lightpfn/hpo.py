from __future__ import annotations

from tabarena.models.lightpfn.model import LightPFNModel
from tabarena.utils.config_utils import ConfigGenerator


# One released checkpoint per task and no tunable training: the default configuration only (four classification or
# two regression members).
def default_config() -> dict:
    """Keep encoded contexts only in the refit model; fold models encode at their single out-of-fold prediction."""
    return {"cache_context": False, "ag.refit_hyperparameters": {"cache_context": True}}


gen_lightpfn = ConfigGenerator(
    model_cls=LightPFNModel,
    search_space={},
    manual_configs=[default_config()],
)

if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    print(
        YamlExperimentSerializer.to_yaml_str(
            experiments=gen_lightpfn.generate_all_bag_experiments(num_random_configs=0),
        ),
    )
