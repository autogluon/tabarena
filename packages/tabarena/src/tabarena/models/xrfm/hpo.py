from __future__ import annotations

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.models.xrfm.model import XRFMModel
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_xrfm(num_random_configs=200) -> list:
    # ConfigSpace is an optional extra; import lazily so simply importing
    # this module (e.g. via the auto-discovery registry) doesn't require it.
    from ConfigSpace import Categorical, ConfigurationSpace, Float

    search_space = ConfigurationSpace(
        space=[
            Float("bandwidth", (0.5, 200), log=True),
            Categorical("standardize_cats", [False]),
            Categorical("bandwidth_mode", ["constant"]),
            Categorical("diag", [False, True]),
            Categorical("early_stop_multiplier", [1.1]),
            Float("exponent", (0.7, 1.4)),
            Float("p_interp", (0.0, 0.8)),
            Categorical("kernel", ["lpq_kermac", "l2"], weights=[0.8, 0.2]),
            Float("reg", (1e-6, 1e1), log=True),
            Categorical("solver", ["solve"]),  # [log_reg] possible for binary classification
            Categorical("classification_mode", ["prevalence"]),
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    return [convert_numpy_dtypes(config) for config in configs]


gen_xrfm = CustomAGConfigGenerator(
    model_cls=XRFMModel,
    search_space_func=generate_configs_xrfm,
    manual_configs=[{}],
)


if __name__ == "__main__":
    from tabarena.benchmark.experiment import YamlExperimentSerializer

    experiments = gen_xrfm.generate_all_bag_experiments(num_random_configs=200)
    YamlExperimentSerializer.to_yaml(
        experiments=experiments,
        path="configs_xrfm.yaml",
    )
