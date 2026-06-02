from __future__ import annotations

from autogluon.tabular.models import XTModel

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_xt(num_random_configs=200):
    # ConfigSpace is an optional extra; import lazily so simply importing
    # this module (e.g. via the auto-discovery registry) doesn't require it.
    from ConfigSpace import Categorical, ConfigurationSpace, Integer

    search_space = ConfigurationSpace(
        space=[
            Categorical("max_features", ["sqrt", 0.5, 0.75, 1.0]),
            Integer("min_samples_split", (2, 32), log=True),
            Categorical(
                "bootstrap", [False]
            ),  # bootstrap=False doesn't allow OOB scores but seems to help
            Categorical(
                "n_estimators", [50]
            ),  # 50 is decent, could go a bit higher for small gains
            Categorical(
                "min_impurity_decrease",
                [0.0, 1e-5, 3e-5, 1e-4, 3e-4, 1e-3],
                weights=[0.5, 0.1, 0.1, 0.1, 0.1, 0.1],
            ),
        ],
        seed=1234,
    )

    configs = search_space.sample_configuration(num_random_configs)
    if num_random_configs == 1:
        configs = [configs]
    configs = [dict(config) for config in configs]
    configs = [convert_numpy_dtypes(config) for config in configs]
    for config in configs:
        if not config["bootstrap"] and "max_samples" in config:
            del config["max_samples"]  # can't be used in this case
        config["ag_args_ensemble"] = {"use_child_oof": False}
    return configs


gen_extratrees = CustomAGConfigGenerator(
    model_cls=XTModel,
    search_space_func=generate_configs_xt,
    manual_configs=[{"ag_args_ensemble": {"use_child_oof": False}}],
)


if __name__ == "__main__":
    print(generate_configs_xt(3))
