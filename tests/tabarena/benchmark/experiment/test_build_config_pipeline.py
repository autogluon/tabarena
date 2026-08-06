"""Tests for the build -> serialize -> load experiment pipeline.

Compute resources, the fold-fitting strategy, and the preprocessing pipeline are
all baked into each Experiment at *build* time by `TabArenaExperimentBundle` (the
preprocessing pipeline + `None` resources are then resolved lazily by the
Experiment itself). Loading is just `YamlExperimentSerializer.from_yaml` (with an
optional `config_index` filter), yielding ready-to-run experiments.
"""

from __future__ import annotations

import copy

import pytest

from tabarena.benchmark.experiment import (
    ModelConstraints,
    TabArenaExperimentBundle,
    YamlExperimentSerializer,
)
from tabarena.benchmark.preprocessing.model_agnostic_default_preprocessing import (
    TabArenaModelAgnosticPreprocessing,
)


def _generate_yaml(
    tmp_path,
    *,
    models,
    n_random_configs: int = 50,
    time_limit: int = 123,
    num_cpus: int | None = 8,
    num_gpus: int = 0,
    memory_limit: int | None = 32,
    time_limit_with_preprocessing: bool = False,
    **bundle_kwargs,
) -> str:
    bundle = TabArenaExperimentBundle(
        n_random_configs=n_random_configs,
        models=models,
        preprocessing_pipelines=["tabarena_default"],
        verbosity=0,
        **bundle_kwargs,
    )
    configs_path = str(tmp_path / "configs.yaml")
    bundle.generate_configs_yaml(
        configs_path=configs_path,
        time_limit=time_limit,
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        memory_limit=memory_limit,
        time_limit_with_preprocessing=time_limit_with_preprocessing,
    )
    return configs_path


def test_build_bakes_resources_fold_fitting_and_carries_preprocessing(tmp_path):
    configs_path = _generate_yaml(
        tmp_path,
        models=[("RealMLP", 0)],
        sequential_local_fold_fitting=True,
        num_cpus=4,
        num_gpus=1,
        memory_limit=16,
    )

    methods = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=None)

    assert len(methods) == 1
    exp = methods[0]
    mk = exp.method_kwargs

    # resources baked into fit_kwargs at build time
    assert mk["fit_kwargs"]["num_cpus"] == 4
    assert mk["fit_kwargs"]["num_gpus"] == 1
    assert mk["fit_kwargs"]["memory_limit"] == 16

    # sequential local fold fitting baked into model hyperparameters at build time
    assert mk["model_hyperparameters"]["ag_args_ensemble"]["fold_fitting_strategy"] == "sequential_local"

    # preprocessing carried as a first-class attribute, applied lazily
    assert exp.preprocessing_pipeline == "tabarena_default"
    assert "feature_generator_cls" not in mk["fit_kwargs"]
    assert "ag.model_specific_feature_generator_kwargs" not in mk["model_hyperparameters"]

    rmk = exp._apply_preprocessing(copy.deepcopy(exp.method_kwargs))
    assert rmk["fit_kwargs"]["feature_generator_cls"] is TabArenaModelAgnosticPreprocessing
    assert rmk["fit_kwargs"]["feature_generator_kwargs"] == {}
    assert "ag.model_specific_feature_generator_kwargs" in rmk["model_hyperparameters"]
    # baked resources + fold fitting are carried through
    assert rmk["fit_kwargs"]["num_cpus"] == 4
    assert rmk["model_hyperparameters"]["ag_args_ensemble"]["fold_fitting_strategy"] == "sequential_local"
    # the original experiment's method_kwargs are left untouched (applied on a copy)
    assert "feature_generator_cls" not in exp.method_kwargs["fit_kwargs"]


def test_build_without_sequential_fold_fitting(tmp_path):
    configs_path = _generate_yaml(
        tmp_path,
        models=[("RealMLP", 0)],
        sequential_local_fold_fitting=False,
        time_limit=60,
        num_cpus=2,
        num_gpus=0,
        memory_limit=8,
    )

    methods = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=None)

    mk = methods[0].method_kwargs
    assert mk["fit_kwargs"]["num_cpus"] == 2
    ag_ensemble = mk["model_hyperparameters"].get("ag_args_ensemble", {})
    assert ag_ensemble.get("fold_fitting_strategy") != "sequential_local"


def test_build_with_none_resources_is_autodetected_lazily(tmp_path):
    # `None` resources are baked as `None` (auto-detect deferred to run time),
    # preserving per-node auto-detection.
    configs_path = _generate_yaml(
        tmp_path,
        models=[("RealMLP", 0)],
        time_limit=60,
        num_cpus=None,
        num_gpus=0,
        memory_limit=None,
    )

    exp = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=None)[0]

    # baked as None on disk / after load
    assert exp.method_kwargs["fit_kwargs"]["num_cpus"] is None
    assert exp.method_kwargs["fit_kwargs"]["memory_limit"] is None

    # resolved lazily to concrete node resources
    resolved = exp._apply_resources(exp.method_kwargs)
    assert isinstance(resolved["fit_kwargs"]["num_cpus"], int)
    assert isinstance(resolved["fit_kwargs"]["memory_limit"], int)


def test_from_yaml_config_index_filters(tmp_path):
    # Two models -> two configs; selecting index [0] returns exactly one.
    configs_path = _generate_yaml(tmp_path, models=[("RealMLP", 0), ("LightGBM", 0)])

    methods = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=[0])

    assert len(methods) == 1


def test_build_bakes_dynamic_validation_protocol_and_round_trips(tmp_path):
    # The bundle default (True) is baked into each experiment and survives YAML round-trip.
    configs_path = _generate_yaml(tmp_path, models=[("RealMLP", 0)])
    exp = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=None)[0]
    assert exp.dynamic_tabarena_validation_protocol is True


def test_build_can_disable_dynamic_validation_protocol(tmp_path):
    configs_path = _generate_yaml(
        tmp_path,
        models=[("RealMLP", 0)],
        dynamic_tabarena_validation_protocol=False,
    )
    exp = YamlExperimentSerializer.from_yaml(path=configs_path, config_index=None)[0]
    assert exp.dynamic_tabarena_validation_protocol is False


def test_bundle_model_constraints_merges_defaults_and_custom():
    custom = ModelConstraints(max_n_features=3)
    bundle = TabArenaExperimentBundle(
        n_random_configs=0,
        models=[],
        preprocessing_pipelines=["default"],
        custom_model_constraints={"MYMODEL": custom},
    )
    effective = bundle.model_constraints
    assert effective["MYMODEL"] is custom  # custom override present
    assert "TABICL" in effective  # default policy preserved


# ---------------------------------------------------------------------------
# tabarena_default applied to a full AutoGluon experiment (AGExperiment)
# ---------------------------------------------------------------------------
_MODEL_SPECIFIC_KEY = "ag.model_specific_feature_generator_kwargs"


def _apply_tabarena_default(fit_kwargs: dict) -> dict:
    """Build an AGExperiment with tabarena_default and return its `_apply_preprocessing` output."""
    from tabarena.benchmark.experiment import AGExperiment

    exp = AGExperiment(name="AutoGluon_x", fit_kwargs=fit_kwargs, preprocessing_pipeline="tabarena_default")
    return exp._apply_preprocessing(copy.deepcopy(exp.method_kwargs))


def _has_model_specific(config) -> bool:
    return isinstance(config, dict) and _MODEL_SPECIFIC_KEY in config


def test_autogluon_experiment_tabarena_default_single_model():
    """A single-model AutoGluon experiment gets the model-agnostic generator + the single model's
    hyperparameters wrapped with the model-specific preprocessing (matching the config path).
    """
    rmk = _apply_tabarena_default({"hyperparameters": {"GBM": {"num_boost_round": 10}}})
    assert rmk["fit_kwargs"]["feature_generator_cls"] is TabArenaModelAgnosticPreprocessing
    assert rmk["fit_kwargs"]["feature_generator_kwargs"] == {}
    assert _has_model_specific(rmk["fit_kwargs"]["hyperparameters"]["GBM"])


def test_autogluon_experiment_tabarena_default_multi_model():
    """Every model (and every config in a per-model list) is wrapped for a multi-model predictor."""
    rmk = _apply_tabarena_default(
        {"hyperparameters": {"GBM": [{"num_boost_round": 10}, {"learning_rate": 0.05}], "CAT": {}, "XGB": {}}},
    )
    assert rmk["fit_kwargs"]["feature_generator_cls"] is TabArenaModelAgnosticPreprocessing
    hp = rmk["fit_kwargs"]["hyperparameters"]
    assert all(_has_model_specific(c) for c in hp["GBM"])  # list of configs, each wrapped
    assert _has_model_specific(hp["CAT"])
    assert _has_model_specific(hp["XGB"])


def _all_wrapped(hyperparameters: dict) -> bool:
    return all(
        _has_model_specific(config)
        for configs in hyperparameters.values()
        for config in (configs if isinstance(configs, list) else [configs])
    )


def test_autogluon_experiment_tabarena_default_bare_preset_resolves_its_hyperparameters():
    """A bare preset run (alias included) takes the `hyperparameters` entry from the preset dict,
    expands it, and wraps every config — same outcome as passing the portfolio explicitly.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    from autogluon.tabular.configs.presets_configs import tabular_presets_dict

    rmk = _apply_tabarena_default({"presets": "extreme"})  # alias of extreme_quality
    hyperparameters = rmk["fit_kwargs"]["hyperparameters"]
    expected = get_hyperparameter_config(tabular_presets_dict["extreme_quality"]["hyperparameters"])
    assert set(hyperparameters) == set(expected)
    assert _all_wrapped(hyperparameters)


def test_autogluon_experiment_tabarena_default_preset_without_hyperparameters_uses_default():
    """A preset that sets no `hyperparameters` (e.g. medium_quality) falls back to AutoGluon's
    `"default"` config, expanded and wrapped — matching what `TabularPredictor.fit` would run.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    rmk = _apply_tabarena_default({"presets": "medium_quality"})
    hyperparameters = rmk["fit_kwargs"]["hyperparameters"]
    assert set(hyperparameters) == set(get_hyperparameter_config("default"))
    assert _all_wrapped(hyperparameters)


def test_autogluon_experiment_tabarena_default_explicit_none_blocks_the_preset():
    """An explicit `hyperparameters=None` key beats the preset's value in AutoGluon
    (`apply_presets` only fills missing keys), so it resolves to `"default"`, not the preset's
    portfolio.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    rmk = _apply_tabarena_default({"presets": "extreme", "hyperparameters": None})
    hyperparameters = rmk["fit_kwargs"]["hyperparameters"]
    assert set(hyperparameters) == set(get_hyperparameter_config("default"))
    assert _all_wrapped(hyperparameters)


def test_autogluon_experiment_tabarena_default_last_preset_wins():
    """With a preset list, the last preset that sets `hyperparameters` wins (AutoGluon's
    first-to-last merge).
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config
    from autogluon.tabular.configs.presets_configs import tabular_presets_dict

    rmk = _apply_tabarena_default({"presets": ["medium_quality", "extreme"]})
    expected = get_hyperparameter_config(tabular_presets_dict["extreme_quality"]["hyperparameters"])
    assert set(rmk["fit_kwargs"]["hyperparameters"]) == set(expected)


def test_autogluon_experiment_tabarena_default_unresolvable_hyperparameters_warns():
    """Hyperparameters of a type the injection cannot handle warn instead of passing silently."""
    with pytest.warns(UserWarning, match="model-specific"):
        rmk = _apply_tabarena_default({"hyperparameters": 123})
    assert rmk["fit_kwargs"]["feature_generator_cls"] is TabArenaModelAgnosticPreprocessing
    assert rmk["fit_kwargs"]["hyperparameters"] == 123  # left untouched for AutoGluon to reject


def test_autogluon_experiment_tabarena_default_named_config_is_expanded():
    """A named AutoGluon config string (what the shipped presets carry, e.g.
    `"noncommercial_2026_08_05"`) is expanded to its config dict and every config is wrapped,
    so a preset-driven run gets the same model-specific preprocessing as a dict-driven run.
    """
    from autogluon.tabular.configs.hyperparameter_configs import get_hyperparameter_config

    rmk = _apply_tabarena_default({"hyperparameters": "very_light"})
    hyperparameters = rmk["fit_kwargs"]["hyperparameters"]
    assert isinstance(hyperparameters, dict)
    assert set(hyperparameters) == set(get_hyperparameter_config("very_light"))
    for configs in hyperparameters.values():
        for config in configs if isinstance(configs, list) else [configs]:
            assert _has_model_specific(config)


def test_autogluon_experiment_tabarena_default_unknown_named_config_raises():
    """A typo in the named config fails at experiment-preprocessing time with the valid names,
    not node-side inside the fit.
    """
    with pytest.raises(ValueError, match="not_a_real_config"):
        _apply_tabarena_default({"hyperparameters": "not_a_real_config"})
