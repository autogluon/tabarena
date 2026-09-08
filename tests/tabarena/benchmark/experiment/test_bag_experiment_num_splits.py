"""`AGModelBagExperiment` and `generate_bag_experiments` with fold counts left to the protocol."""

from __future__ import annotations

import pytest
from autogluon.tabular.models import LGBModel

from tabarena.benchmark.experiment import AGModelBagExperiment
from tabarena.benchmark.task.metadata import AUTO_NUM_SPLITS, ValidationMetadata
from tabarena.utils.config_utils import _apply_seed_to_bag_configs, generate_bag_experiments


def test_bag_experiment_defaults_to_auto_counts():
    exp = AGModelBagExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={})
    fit_kwargs = exp.method_kwargs["fit_kwargs"]
    assert fit_kwargs["num_bag_folds"] == AUTO_NUM_SPLITS
    assert fit_kwargs["num_bag_sets"] == AUTO_NUM_SPLITS


def test_bag_experiment_keeps_explicit_counts():
    exp = AGModelBagExperiment(
        name="lgb", model_cls=LGBModel, model_hyperparameters={}, num_bag_folds=2, num_bag_sets=3
    )
    fit_kwargs = exp.method_kwargs["fit_kwargs"]
    assert fit_kwargs["num_bag_folds"] == 2
    assert fit_kwargs["num_bag_sets"] == 3


@pytest.mark.parametrize("kwargs", [{"num_bag_folds": 1}, {"num_bag_sets": 0}, {"num_bag_folds": "eight"}])
def test_bag_experiment_rejects_invalid_counts(kwargs):
    with pytest.raises(AssertionError):
        AGModelBagExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={}, **kwargs)


def test_bag_experiment_auto_counts_place_time_limit_on_the_bag():
    """An "auto" count is a bagged fit, so the limit goes under `ag_args_ensemble` as for any k > 1."""
    exp = AGModelBagExperiment(name="lgb", model_cls=LGBModel, model_hyperparameters={}, time_limit=60)
    hp = exp.method_kwargs["model_hyperparameters"]
    assert hp["ag_args_ensemble"]["ag.max_time_limit"] == 60
    assert "ag.max_time_limit" not in hp


def test_generate_bag_experiments_defaults_to_auto_counts():
    (exp,) = generate_bag_experiments(model_cls=LGBModel, configs=[{}], time_limit=60)
    assert exp.method_kwargs["fit_kwargs"]["num_bag_folds"] == AUTO_NUM_SPLITS
    assert exp.method_kwargs["fit_kwargs"]["num_bag_sets"] == AUTO_NUM_SPLITS


def test_seed_offsets_treat_auto_counts_as_the_defaults():
    defaults = ValidationMetadata()
    auto = _apply_seed_to_bag_configs(
        [{}, {}], "fold-config-wise", num_bag_folds=AUTO_NUM_SPLITS, num_bag_sets=AUTO_NUM_SPLITS
    )
    explicit = _apply_seed_to_bag_configs(
        [{}, {}],
        "fold-config-wise",
        num_bag_folds=defaults.default_num_folds,
        num_bag_sets=defaults.default_num_repeats,
    )
    assert auto == explicit
    assert auto[1]["ag_args_ensemble"]["model_random_seed"] == defaults.default_num_folds * defaults.default_num_repeats
