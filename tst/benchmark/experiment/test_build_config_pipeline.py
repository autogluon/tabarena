"""Tests for the build -> serialize -> load experiment pipeline.

Compute resources, the fold-fitting strategy, and the preprocessing pipeline are
all baked into each Experiment at *build* time by `TabArenaExperimentBundle` (the
preprocessing pipeline + `None` resources are then resolved lazily by the
Experiment itself). Loading is just `YamlExperimentSerializer.from_yaml` (with an
optional `config_index` filter), yielding ready-to-run experiments.
"""

from __future__ import annotations

import copy

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
