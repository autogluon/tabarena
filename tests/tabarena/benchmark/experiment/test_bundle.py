"""Pin the resolved field values of the experiment bundle subclasses.

These tests instantiate the child bundles with no arguments and assert every
post-init attribute against a hardcoded expectation. The intent is to catch
*accidental* changes to the parent `TabArenaExperimentBundle` defaults (or to a
child override) when someone edits the dataclass in the future: if a default
moves, or a field is added/removed, exactly one of these tests fails and points
at the drift. Update the expected dict deliberately when the change is intended.
"""

from __future__ import annotations

import dataclasses

import pytest

from tabarena.benchmark.experiment import (
    BeyondArenaExperimentBundle,
    TabArenaExperimentBundle,
    TabArenaV0pt1ExperimentBundle,
)
from tabarena.benchmark.experiment.bundle import (
    TABICL_CONSTRAINTS,
    TABPFNV2_CONSTRAINTS,
)

# The complete set of instance fields the parent declares. Pinned so that adding
# or removing a field on the parent forces a conscious update to these tests
# (and a re-check of the per-subclass expectations below).
EXPECTED_FIELD_NAMES = {
    "models",
    "n_random_configs",
    "preprocessing_pipelines",
    "model_agnostic_preprocessing",
    "max_predict_batch_size",
    "sequential_local_fold_fitting",
    "outer_experiments",
    "holdout_experiments",
    "system_experiments",
    "model_artifacts_base_path",
    "verbosity",
    "model_verbosity",
    "adapt_num_folds_to_n_classes",
    "shuffle_features",
    "dynamic_tabarena_validation_protocol",
    "text_cache_mode",
    "custom_model_constraints",
}

# The defaults shared by every subclass (i.e. inherited from the parent and not
# overridden). Subclass-specific expectations below extend this baseline.
COMMON_INHERITED_DEFAULTS = {
    "models": [],
    "model_agnostic_preprocessing": True,
    "max_predict_batch_size": None,
    "sequential_local_fold_fitting": False,
    "outer_experiments": False,
    "holdout_experiments": False,
    "system_experiments": False,
    "model_artifacts_base_path": "/tmp",  # noqa: S108
    "verbosity": 2,
    "model_verbosity": 4,
    "custom_model_constraints": {},
}

# Hardcoded full post-init state for each subclass instantiated with no args.
BEYOND_ARENA_EXPECTED = {
    **COMMON_INHERITED_DEFAULTS,
    "n_random_configs": 25,
    "preprocessing_pipelines": ["tabarena_default"],
    "adapt_num_folds_to_n_classes": True,
    "shuffle_features": True,
    "dynamic_tabarena_validation_protocol": True,
    "text_cache_mode": "require",
}

TABARENA_V0PT1_EXPECTED = {
    **COMMON_INHERITED_DEFAULTS,
    "n_random_configs": 200,
    "preprocessing_pipelines": ["default"],
    "adapt_num_folds_to_n_classes": False,
    "shuffle_features": False,
    "dynamic_tabarena_validation_protocol": False,
    "text_cache_mode": "off",
}


def _fields_as_dict(bundle: TabArenaExperimentBundle) -> dict:
    return {f.name: getattr(bundle, f.name) for f in dataclasses.fields(bundle)}


def test_parent_field_set_is_unchanged():
    """Guard against fields being silently added to / removed from the parent."""
    actual = {f.name for f in dataclasses.fields(TabArenaExperimentBundle)}
    assert actual == EXPECTED_FIELD_NAMES


def test_beyond_arena_bundle_post_init_values():
    bundle = BeyondArenaExperimentBundle()
    assert _fields_as_dict(bundle) == BEYOND_ARENA_EXPECTED


def test_tabarena_v0pt1_bundle_post_init_values():
    bundle = TabArenaV0pt1ExperimentBundle()
    assert _fields_as_dict(bundle) == TABARENA_V0PT1_EXPECTED


def test_subclass_field_sets_match_parent():
    """Subclasses must not introduce or drop fields relative to the parent."""
    for cls in (BeyondArenaExperimentBundle, TabArenaV0pt1ExperimentBundle):
        assert {f.name for f in dataclasses.fields(cls)} == EXPECTED_FIELD_NAMES


def test_default_model_constraints_are_shared_across_subclasses():
    """The class-level constraint map is inherited unchanged by both subclasses."""
    expected = {
        "TABICL": TABICL_CONSTRAINTS,
        "TA-TABICL": TABICL_CONSTRAINTS,
        "TABPFNV2": TABPFNV2_CONSTRAINTS,
        "TA-TABPFNV2": TABPFNV2_CONSTRAINTS,
        "MITRA": TABPFNV2_CONSTRAINTS,
    }
    for cls in (TabArenaExperimentBundle, BeyondArenaExperimentBundle, TabArenaV0pt1ExperimentBundle):
        assert expected == cls.DEFAULT_MODEL_CONSTRAINTS

    # With no custom overrides, the effective `model_constraints` equals the defaults.
    assert BeyondArenaExperimentBundle().model_constraints == expected
    assert TabArenaV0pt1ExperimentBundle().model_constraints == expected


def test_build_experiments_attaches_model_constraints():
    """build_experiments resolves each experiment's constraints by AG key and attaches them."""
    from autogluon.tabular.models import LGBModel

    from tabarena.benchmark.experiment import AGModelBagExperiment, ModelConstraints

    gbm_constraints = ModelConstraints(max_n_samples_train_per_fold=123)
    explicit = ModelConstraints(max_n_features=7)
    bundle = TabArenaExperimentBundle(
        models=[
            # Built via the registry: LGBModel's AG key is "GBM" -> gets gbm_constraints.
            ("LightGBM", 0),
            # Pre-built passthrough with explicit constraints: kept, not overridden.
            AGModelBagExperiment(
                name="explicitly_constrained",
                model_cls=LGBModel,
                model_hyperparameters={},
                num_bag_folds=2,
                time_limit=60,
                model_constraints=explicit,
            ),
        ],
        n_random_configs=0,
        preprocessing_pipelines=["default"],
        custom_model_constraints={"GBM": gbm_constraints},
    )
    experiments = bundle.build_experiments(
        time_limit=60,
        num_cpus=1,
        num_gpus=0,
        memory_limit=4,
        time_limit_with_preprocessing=False,
    )
    by_name = {experiment.name: experiment for experiment in experiments}
    assert by_name["explicitly_constrained"].model_constraints == explicit
    registry_built = [e for name, e in by_name.items() if name != "explicitly_constrained"]
    assert registry_built, "expected at least the default LightGBM config"
    assert all(e.model_constraints == gbm_constraints for e in registry_built)


# ===========================================================================
# Per-model hyperparameter override — the optional 3rd `models` tuple element
# ===========================================================================


def _model_hyperparameters(experiment):
    """Read the per-model hyperparameters off a built experiment, regardless of flavour.

    Bagged / holdout experiments store them under ``model_hyperparameters``; the no-validation
    outer flavour stores them under ``hyperparameters``.
    """
    method_kwargs = experiment.method_kwargs
    return method_kwargs.get("model_hyperparameters", method_kwargs.get("hyperparameters"))


def _build_single_lightgbm(*, hyperparameters, **bundle_kwargs):
    """Build the single default-LightGBM experiment with a per-model hyperparameter override."""
    bundle = BeyondArenaExperimentBundle(
        models=[("LightGBM", 0, hyperparameters)],
        **bundle_kwargs,
    )
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    assert len(experiments) == 1
    return experiments[0]


@pytest.mark.parametrize(
    "bundle_kwargs",
    [{}, {"holdout_experiments": True}, {"outer_experiments": True}],
    ids=["bagged", "holdout", "outer"],
)
def test_per_model_hyperparameters_injected_for_each_flavour(bundle_kwargs):
    """The 3rd tuple element lands in the model hyperparameters for bag / holdout / outer alike."""
    experiment = _build_single_lightgbm(hyperparameters={"num_boost_round": 100}, **bundle_kwargs)
    hyperparameters = _model_hyperparameters(experiment)
    assert hyperparameters["num_boost_round"] == 100
    # The bundle-level extras (here model_verbosity -> ag.verbosity) are still merged in alongside.
    assert hyperparameters["ag.verbosity"] == BeyondArenaExperimentBundle.model_verbosity


def test_per_model_hyperparameters_supports_multiple_keys():
    """Arbitrary hyperparameters (not just one) flow through the 3rd tuple element."""
    experiment = _build_single_lightgbm(hyperparameters={"learning_rate": 0.05, "num_leaves": 31})
    hyperparameters = _model_hyperparameters(experiment)
    assert hyperparameters["learning_rate"] == 0.05
    assert hyperparameters["num_leaves"] == 31


def test_per_model_hyperparameters_only_apply_to_that_model():
    """An override on one model must not leak onto the other models in the bundle."""
    bundle = BeyondArenaExperimentBundle(
        models=[("LightGBM", 0, {"num_boost_round": 100}), ("RandomForest", 0)],
    )
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    by_name = {e.name: e for e in experiments}
    lightgbm = next(e for name, e in by_name.items() if name.startswith("LightGBM"))
    random_forest = next(e for name, e in by_name.items() if name.startswith("RandomForest"))
    assert _model_hyperparameters(lightgbm)["num_boost_round"] == 100
    assert "num_boost_round" not in _model_hyperparameters(random_forest)


def test_two_tuple_still_works_without_override():
    """Backward compatibility: a plain (name, n_configs) entry builds as before (no override)."""
    bundle = BeyondArenaExperimentBundle(models=[("LightGBM", 0)])
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    assert len(experiments) == 1
    assert "num_boost_round" not in _model_hyperparameters(experiments[0])


def test_per_model_hyperparameters_rejected_for_full_autogluon_entry():
    """Per-model hyperparameters are not supported for full ``AutoGluon...`` entries."""
    bundle = BeyondArenaExperimentBundle(models=[("AutoGluon_bq", {}, {"num_boost_round": 100})])
    with pytest.raises(ValueError, match="not supported for full"):
        bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)


# ===========================================================================
# Full AutoGluon experiment entries — non-IID validation protocol forwarding
# ===========================================================================


def _build_single_autogluon(*, agexp_kwargs, **bundle_kwargs):
    from tabarena.benchmark.experiment import AGExperiment

    bundle = BeyondArenaExperimentBundle(models=[("AutoGluon_custom", agexp_kwargs)], **bundle_kwargs)
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    assert len(experiments) == 1
    exp = experiments[0]
    assert isinstance(exp, AGExperiment)
    return exp


def test_autogluon_experiment_inherits_dynamic_validation_protocol():
    """A full AutoGluon experiment gets the bundle's non-IID validation protocol auto-forwarded."""
    exp = _build_single_autogluon(agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}, "num_bag_folds": 8}})
    # BeyondArena defaults the protocol on; the AutoGluon path now forwards it (was previously lost).
    assert exp.dynamic_tabarena_validation_protocol is True
    # Custom hyperparameters and bagging settings pass through to the predictor fit kwargs.
    assert exp.method_kwargs["fit_kwargs"]["hyperparameters"] == {"GBM": {}}
    assert exp.method_kwargs["fit_kwargs"]["num_bag_folds"] == 8


def test_autogluon_experiment_protocol_override_is_respected():
    """An explicit per-entry ``dynamic_tabarena_validation_protocol`` wins over the bundle default."""
    exp = _build_single_autogluon(
        agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}}, "dynamic_tabarena_validation_protocol": False},
    )
    assert exp.dynamic_tabarena_validation_protocol is False


def test_autogluon_experiment_protocol_off_when_bundle_disables_it():
    """When the bundle disables the protocol, the AutoGluon experiment does not get it."""
    exp = _build_single_autogluon(
        agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}}},
        dynamic_tabarena_validation_protocol=False,
    )
    assert exp.dynamic_tabarena_validation_protocol is False


def test_autogluon_experiment_inherits_preprocessing_pipeline():
    """A full AutoGluon experiment inherits the bundle's preprocessing pipeline (BeyondArena uses
    tabarena_default), so it gets the same preprocessing as the config experiments.
    """
    exp = _build_single_autogluon(agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}}})
    assert exp.preprocessing_pipeline == "tabarena_default"


def test_autogluon_experiment_preprocessing_override_is_respected():
    """An explicit per-entry preprocessing_pipeline wins over the bundle default."""
    exp = _build_single_autogluon(
        agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}}, "preprocessing_pipeline": "default"},
    )
    assert exp.preprocessing_pipeline == "default"
