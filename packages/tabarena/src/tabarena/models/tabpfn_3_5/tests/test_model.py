from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(("module", "model"), [("tabpfn_3_5", "TabPFN35Model"), ("tabpfn_3", "TabPFN3Model")])
def test_the_model_keeps_one_seed_across_a_bags_folds_by_default(module, model):
    model_cls = getattr(importlib.import_module(f"tabarena.models.{module}.model"), model)
    assert model_cls._get_default_ag_args_ensemble()["vary_seed_across_folds"] is False


@pytest.mark.parametrize(("module", "generator"), [("tabpfn_3_5", "gen_tabpfn_3_5"), ("tabpfn_3", "gen_tabpfn_3")])
def test_the_benchmark_seed_policy_still_decides_for_its_experiments(module, generator):
    gen = getattr(importlib.import_module(f"tabarena.models.{module}.hpo"), generator)
    # the bundles' seed policy stamps its value on every bagged config, which overrides the model's default
    experiments = gen.generate_all_bag_experiments(num_random_configs=0, add_seed="fold-config-wise")
    assert experiments
    for experiment in experiments:
        ensemble = experiment.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]
        assert ensemble["vary_seed_across_folds"] is True
