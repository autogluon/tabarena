from __future__ import annotations

import importlib

import pytest


@pytest.mark.parametrize(("module", "generator"), [("tabpfn_3_5", "gen_tabpfn_3_5"), ("tabpfn_3", "gen_tabpfn_3")])
def test_a_bag_keeps_one_seed_across_its_folds(module, generator):
    gen = getattr(importlib.import_module(f"tabarena.models.{module}.hpo"), generator)
    # the official bundles' seed policy varies the seed across a bag's folds
    experiments = gen.generate_all_bag_experiments(num_random_configs=0, add_seed="fold-config-wise")
    assert experiments
    for experiment in experiments:
        ensemble = experiment.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]
        assert ensemble["vary_seed_across_folds"] is False
