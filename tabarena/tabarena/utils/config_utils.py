from __future__ import annotations

import copy
from typing import Type, Literal

from autogluon.core.searcher.local_random_searcher import LocalRandomSearcher
from autogluon.core.models import AbstractModel

from tabarena.benchmark.experiment import AGModelBagExperiment, AGModelExperiment


def configs_to_name_dict(configs, name_prefix, model_type):
    configs_dict = {}
    for c in configs:
        name_suffix = c["ag_args"]["name_suffix"]
        c_name = f'{name_prefix}{name_suffix}'
        config_dict = dict()
        config_dict['hyperparameters'] = c
        config_dict['name_prefix'] = name_prefix
        config_dict['name_suffix'] = name_suffix
        config_dict['model_type'] = model_type
        configs_dict[c_name] = config_dict
    return configs_dict


def combine_manual_and_random_configs(manual_configs, random_configs, name_id_suffix: str = ""):
    combined_configs = []
    for i, config in enumerate(manual_configs):
        combined_configs.append(add_suffix_to_config(config=config, suffix=f'_c{i+1}{name_id_suffix}'))
    for i, config in enumerate(random_configs):
        combined_configs.append(add_suffix_to_config(config=config, suffix=f'_r{i+1}{name_id_suffix}'))
    return combined_configs


def add_suffix_to_config(config, suffix):
    if "ag_args" in config:
        raise AssertionError(f"ag_args already exists in config!")
    config = copy.deepcopy(config)
    config['ag_args'] = {'name_suffix': suffix}
    return config

def add_seed_logic(config: dict, random_seed: int, vary_seed_across_folds: bool) -> dict:
    config = copy.deepcopy(config)
    if "ag_args_ensemble" not in config:
        config["ag_args_ensemble"] = {}
    config["ag_args_ensemble"]["model_random_seed"] = random_seed
    config["ag_args_ensemble"]["vary_seed_across_folds"] = vary_seed_across_folds
    return config

def add_fold_fitting_strategy(config: dict, fold_fitting_strategy: str) -> dict:
    config = copy.deepcopy(config)
    if "ag_args_ensemble" not in config:
        config["ag_args_ensemble"] = {}
    config["ag_args_ensemble"]["fold_fitting_strategy"] = fold_fitting_strategy
    return config

def get_random_searcher(search_space):
    searcher = LocalRandomSearcher(search_space=search_space)
    searcher.get_config()  # Clean out default
    return searcher


class AbstractConfigGenerator:
    def __init__(
        self,
        manual_configs: list[dict] | None = None,
    ):
        if manual_configs is None:
            manual_configs = []
        self.manual_configs = manual_configs


class AGConfigGenerator(AbstractConfigGenerator):
    def __init__(
        self,
        name: str,
        model_type: str,
        model_cls,
        manual_configs: list[dict] | None = None,
    ):
        super().__init__(manual_configs=manual_configs)
        self.name = name
        self.model_type = model_type
        self.model_cls = model_cls

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        raise NotImplementedError

    def generate_all_configs(self, num_random_configs):
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs)
        configs_dict = configs_to_name_dict(configs=configs, name_prefix=self.name, model_type=self.model_type)
        return configs_dict

    def generate_all_configs_lst(self, num_random_configs: int, name_id_suffix: str = "") -> list[dict]:
        if num_random_configs > 0:
            random_configs = self.get_searcher_configs(num_random_configs)
        else:
            random_configs = []
        configs = combine_manual_and_random_configs(manual_configs=self.manual_configs, random_configs=random_configs, name_id_suffix=name_id_suffix)
        return configs

    def generate_all_bag_experiments(
        self,
        num_random_configs: int,
        name_id_suffix: str = "",
        add_seed: Literal["static", "fold-wise", "fold-config-wise"] = "static",
        method_kwargs: dict | None = None,
        fold_fitting_strategy: None | Literal["sequential_local"] = None,
        **kwargs,
    ) -> list:
        """Generate experiments with bagging models for the search space.

        Parameters
        ----------
        num_random_configs : int
            The number of random configurations to generate.
        name_id_suffix: str
            A suffix to append to the names of the configuration. Use this to
            distinguish between different runs or configurations of the same model.
        add_seed: {"static", "fold-wise", "fold-config-wise"}
            How to handle random seeds in the configurations.
                - "static": Use a fixed seed across all folds.
                - "fold-wise": Vary the side acros folds
                - "fold-config-wise": Vary the seed across folds and configurations.
        method_kwargs: dict | None
            Additional keyword arguments to pass to the `generate_bag_experiments`
            function. For example, you can modify the init kwargs of TabularPredictor
            runner by `method_kwargs=dict(init_kwargs=dict(path="./my_custom_path"))`
        fold_fitting_strategy: None | {"sequential_local"}
            If specified, this fitting strategy will be added to each configuration.
            It is highly recommended to use "sequential_local" when testing or running
            locally without good CPUs for parallelism.
        """
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs, name_id_suffix=name_id_suffix)
        experiments = generate_bag_experiments(
            model_cls=self.model_cls,
            configs=configs,
            name_suffix_from_ag_args=True,
            add_seed=add_seed,
            method_kwargs=method_kwargs,
            fold_fitting_strategy=fold_fitting_strategy,
            **kwargs
        )
        return experiments

    def generate_all_holdout_experiments(
        self,
        num_random_configs: int,
        name_id_suffix: str = "",
        method_kwargs: dict | None = None,
        **kwargs,
    ) -> list:
        """Generate experiments with holdout models for the search space.

        Parameters
        ----------
        num_random_configs : int
            The number of random configurations to generate.
        name_id_suffix: str
            A suffix to append to the names of the configuration. Use this to
            distinguish between different runs or configurations of the same model.
        method_kwargs: dict | None
            Additional keyword arguments to pass to the `generate_bag_experiments`
            function. For example, you can modify the init kwargs of TabularPredictor
            runner by `method_kwargs=dict(init_kwargs=dict(path="./my_custom_path"))`
        """
        configs = self.generate_all_configs_lst(num_random_configs=num_random_configs, name_id_suffix=name_id_suffix)
        experiments = generate_holdout_experiments(
            model_cls=self.model_cls,
            configs=configs,
            name_suffix_from_ag_args=True,
            method_kwargs=method_kwargs,
            **kwargs
        )
        return experiments


class ConfigGenerator(AGConfigGenerator):
    def __init__(
        self,
        search_space: dict,
        model_cls: Type[AbstractModel],
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        if name is None:
            name = model_cls.ag_name
        model_type = model_cls.ag_key
        assert name is not None, "set `ag_name` and `ag_key` in the model class!"
        super().__init__(
            name=name,
            model_type=model_type,
            model_cls=model_cls,
            manual_configs=manual_configs,
        )
        self.search_space = search_space

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        searcher = get_random_searcher(self.search_space)
        return [searcher.get_config() for _ in range(num_configs)]


class CustomAGConfigGenerator(AGConfigGenerator):
    def __init__(
        self,
        model_cls: Type[AbstractModel],
        search_space_func,
        name: str | None = None,
        manual_configs: list[dict] | None = None,
    ):
        if name is None:
            name = model_cls.ag_name
        model_type = model_cls.ag_key
        super().__init__(
            name=name,
            model_type=model_type,
            model_cls=model_cls,
            manual_configs=manual_configs,
        )
        self.search_space_func = search_space_func
        self.model_cls = model_cls

    def get_searcher_configs(self, num_configs: int) -> list[dict]:
        return self.search_space_func(num_configs)


def generate_bag_experiments(
    model_cls,
    configs: list[dict],
    time_limit: float | None = 3600,
    num_bag_folds: int = 8,
    num_bag_sets: int = 1,
    name_suffix_from_ag_args: bool = False,
    name_id_prefix: str = "r",
    name_id_suffix: str = "",
    name_bag_suffix: str = "_BAG_L1",
    add_name_suffix_to_params: bool = True,
    add_seed: Literal["static", "fold-wise", "fold-config-wise"] = "static",
    fold_fitting_strategy: None | Literal["sequential_local"] = None,
    **kwargs,
) -> list[AGModelBagExperiment]:
    experiments = []
    if kwargs is None:
        kwargs = {}

    if add_seed == "static":
        configs = [
            add_seed_logic(config=config, random_seed=0, vary_seed_across_folds=False)
            for config in configs
        ]
    elif add_seed == "fold-wise":
        configs = [
            add_seed_logic(config=config, random_seed=0, vary_seed_across_folds=True)
            for config in configs
        ]
    elif add_seed == "fold-config-wise":
        offset_between_configs = num_bag_sets * num_bag_folds
        configs = [
            add_seed_logic(config=config, random_seed=i * offset_between_configs, vary_seed_across_folds=True)
            for i, config in enumerate(configs)
        ]
    else:
        raise ValueError(
            f"Invalid add_seed value: {add_seed}. Choose from 'static', 'fold-wise', or 'fold-config-wise'."
        )
    if fold_fitting_strategy is not None:
        configs = [
            add_fold_fitting_strategy(config=config, fold_fitting_strategy=fold_fitting_strategy)
            for config in configs
        ]

    for i, config in enumerate(configs):
        if name_suffix_from_ag_args:
            name_suffix = config.get("ag_args", {}).get("name_suffix", "")
        else:
            name_suffix = f"_{name_id_prefix}{i+1}{name_id_suffix}"
            if add_name_suffix_to_params:
                config = add_suffix_to_config(config=config, suffix=name_suffix)
        name = f"{model_cls.ag_name}{name_suffix}{name_bag_suffix}"
        experiment = AGModelBagExperiment(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=config,
            num_bag_folds=num_bag_folds,
            num_bag_sets=num_bag_sets,
            time_limit=time_limit,
            **kwargs,
        )
        experiments.append(experiment)
    return experiments


def generate_holdout_experiments(
    model_cls,
    configs: list[dict],
    time_limit: float | None = 3600,
    name_suffix_from_ag_args: bool = False,
    name_id_prefix: str = "r",
    name_id_suffix: str = "",
    add_name_suffix_to_params: bool = True,
    **kwargs,
) -> list[AGModelBagExperiment]:
    experiments = []
    if kwargs is None:
        kwargs = {}

    for i, config in enumerate(configs):
        if name_suffix_from_ag_args:
            name_suffix = config.get("ag_args", {}).get("name_suffix", "")
        else:
            name_suffix = f"_{name_id_prefix}{i+1}{name_id_suffix}"
            if add_name_suffix_to_params:
                config = add_suffix_to_config(config=config, suffix=name_suffix)
        name = f"{model_cls.ag_name}{name_suffix}"
        experiment = AGModelExperiment(
            name=name,
            model_cls=model_cls,
            model_hyperparameters=config,
            time_limit=time_limit,
            **kwargs,
        )
        experiments.append(experiment)
    return experiments
