from __future__ import annotations

from autogluon.tabular.models import RFModel

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_rf(num_random_configs=200):
    # ConfigSpace is an optional extra; import lazily so simply importing
    # this module (e.g. via the auto-discovery registry) doesn't require it.
    from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

    search_space = ConfigurationSpace(
        space=[
            Float("max_features", (0.4, 1.0)),
            Float("max_samples", (0.5, 1.0)),
            Integer("min_samples_split", (2, 4), log=True),
            Categorical(
                "bootstrap", [False, True]
            ),  # bootstrap=False doesn't allow OOB scores but seems to help
            Categorical(
                "n_estimators", [50]
            ),  # 50 is decent, could go a bit higher for small gains
            Float("min_impurity_decrease", (1e-5, 1e-3), log=True),
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    configs = [convert_numpy_dtypes(config) for config in configs]
    for config in configs:
        if not config["bootstrap"]:
            del config["max_samples"]  # can't be used in this case
        config["ag_args_ensemble"] = {"use_child_oof": False}
    return configs


gen_randomforest = CustomAGConfigGenerator(
    model_cls=RFModel,
    search_space_func=generate_configs_rf,
    manual_configs=[{}],
)


if __name__ == "__main__":
    print(generate_configs_rf(3))
