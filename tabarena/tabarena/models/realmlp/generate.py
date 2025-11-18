from __future__ import annotations

import numpy as np

from tabarena.benchmark.experiment import YamlExperimentSerializer
from tabarena.benchmark.models.ag.realmlp.realmlp_model import RealMLPModel
from tabarena.models.utils import convert_numpy_dtypes
from tabarena.utils.config_utils import CustomAGConfigGenerator, generate_bag_experiments


def generate_single_config_realmlp(rng):
    # common search space
    params = {
        "n_hidden_layers": rng.integers(2, 4, endpoint=True),
        "hidden_sizes": "rectangular",
        "hidden_width": rng.choice([256, 384, 512]),
        "p_drop": rng.uniform(0.0, 0.5),
        "act": "mish",
        "plr_sigma": np.exp(rng.uniform(np.log(1e-2), np.log(50))),
        "sq_mom": 1.0 - np.exp(rng.uniform(np.log(5e-3), np.log(5e-2))),
        "plr_lr_factor": np.exp(rng.uniform(np.log(5e-2), np.log(3e-1))),
        "scale_lr_factor": np.exp(rng.uniform(np.log(2.0), np.log(10.0))),
        "first_layer_lr_factor": np.exp(rng.uniform(np.log(0.3), np.log(1.5))),
        "ls_eps_sched": "coslog4",
        "ls_eps": np.exp(rng.uniform(np.log(5e-3), np.log(1e-1))),
        "p_drop_sched": "flat_cos",
        "lr": np.exp(rng.uniform(np.log(2e-2), np.log(3e-1))),
        "wd": np.exp(rng.uniform(np.log(1e-3), np.log(5e-2))),
        "use_ls": rng.choice(
            [False, True]  # changed "auto" to False, to have it equal for all metrics
        ),  # use label smoothing (will be ignored for regression)
        "max_one_hot_cat_size": np.floor(np.exp(rng.uniform(np.log(4.0), np.log(33.0)))).item(),
        "embedding_size": rng.choice([4, 8, 16]),
        "n_ens": 8,  # 16 might still be feasible
        "ens_av_before_softmax": False,
    }

    if rng.uniform(0.0, 1.0) > 0.5:
        # large configs
        params["plr_hidden_1"] = rng.choice([8, 16, 32, 64])
        params["plr_hidden_2"] = rng.choice([8, 16, 32, 64])
        params["n_epochs"] = rng.choice([256, 512])
        params["use_early_stopping"] = True
    else:
        # default values, used here to always set the same set of parameters
        params["plr_hidden_1"] = 16
        params["plr_hidden_2"] = 4
        params["n_epochs"] = 256
        params["use_early_stopping"] = False

    return convert_numpy_dtypes(params)


def generate_configs_realmlp(num_random_configs=200, seed=1234):
    # note: this doesn't set val_metric_name, which should be set outside
    rng = np.random.default_rng(seed)
    return [generate_single_config_realmlp(rng) for _ in range(num_random_configs)]


gen_realmlp = CustomAGConfigGenerator(
    model_cls=RealMLPModel,
    search_space_func=generate_configs_realmlp,
    manual_configs=[{}],
)


if __name__ == "__main__":
    configs_yaml = []
    config_defaults = [{}]
    configs = generate_configs_realmlp(100, seed=1234)

    experiments_realmlp_streamlined = gen_realmlp.generate_all_bag_experiments(100)

    experiments_default = generate_bag_experiments(
        model_cls=RealMLPModel,
        configs=config_defaults,
        time_limit=3600,
        name_id_prefix="c",
    )
    experiments_random = generate_bag_experiments(
        model_cls=RealMLPModel, configs=configs, time_limit=3600
    )
    experiments = experiments_default + experiments_random
    YamlExperimentSerializer.to_yaml(
        experiments=experiments, path="configs_realmlp.yaml"
    )
