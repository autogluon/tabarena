from __future__ import annotations

import pytest


@pytest.mark.parametrize("module", ["tabpfn_3_5", "tabpfn_3"])
def test_every_model_of_a_bag_gets_the_fixed_seed(tmp_path, module):
    import importlib

    name = {"tabpfn_3_5": "TabPFN35Model", "tabpfn_3": "TabPFN3Model"}[module]
    model_cls = getattr(importlib.import_module(f"tabarena.models.{module}.model"), name)
    # AutoGluon passes each fold model of a bag its own seed (`vary_seed_across_folds`)
    for fold_seed in (0, 3):
        model = model_cls(path=str(tmp_path), name=f"fold{fold_seed}", hyperparameters={})
        model.initialize(random_seed=fold_seed)
        assert model.random_seed == model_cls.fixed_random_state == 0
        assert model._get_model_params()[model.seed_name] == 0
