from __future__ import annotations

from autogluon.tabular.models import RFModel
from ConfigSpace import Categorical, ConfigurationSpace, Float, Integer

from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator


def generate_configs_rf(num_random_configs=200):
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
            # for raw model performance it seems to be best to not tune this
            # Integer('min_samples_leaf', (1, 4), log=True),
            # could set max_depth or max_leaf_nodes to reduce compute overhead, but it seems to be slightly worse
            # Categorical('max_depth', [16, 20, None]),
            # Integer('max_leaf_nodes', (5000, 50_000), use_log=True),
            # this seems to be quite helpful although including zero in the space might be helpful as well
            Float("min_impurity_decrease", (1e-5, 1e-3), log=True),
            # for regression it is important to standardize targets for min_impurity_decrease>0
            # having values > 0 can also speed up things
            # we could tune something like max_one_hot_cat_size as well if we had this implemented,
            # or target encoding etc.
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
