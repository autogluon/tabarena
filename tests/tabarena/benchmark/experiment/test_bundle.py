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
    ValidationProtocol,
)
from tabarena.benchmark.experiment.bundle import (
    TABICL_CONSTRAINTS,
    TABPFNV2_CONSTRAINTS,
)
from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)

# The complete set of instance fields the parent declares. Pinned so that adding
# or removing a field on the parent forces a conscious update to these tests
# (and a re-check of the per-subclass expectations below).
EXPECTED_FIELD_NAMES = {
    "models",
    "n_random_configs",
    "default_seed_config",
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
    "model_verbosity_overrides",
    "shuffle_features",
    "validation_protocol",
    "text_cache_mode",
    "custom_model_constraints",
}

# The defaults shared by every subclass (i.e. inherited from the parent and not
# overridden). Subclass-specific expectations below extend this baseline.
COMMON_INHERITED_DEFAULTS = {
    "models": [],
    "default_seed_config": "fold-config-wise",
    "model_agnostic_preprocessing": True,
    "max_predict_batch_size": None,
    "sequential_local_fold_fitting": False,
    "outer_experiments": False,
    "holdout_experiments": False,
    "system_experiments": False,
    "model_artifacts_base_path": "/tmp",  # noqa: S108
    "verbosity": 2,
    "model_verbosity": 4,
    "model_verbosity_overrides": {"CatBoost": 2},
    "validation_protocol": None,
    "custom_model_constraints": {},
}

# Hardcoded full post-init state for each subclass instantiated with no args.
BEYOND_ARENA_EXPECTED = {
    **COMMON_INHERITED_DEFAULTS,
    "n_random_configs": 25,
    "preprocessing_pipelines": ["tabarena_default"],
    "shuffle_features": True,
    "text_cache_mode": "require",
}

TABARENA_V0PT1_EXPECTED = {
    **COMMON_INHERITED_DEFAULTS,
    "n_random_configs": 200,
    "preprocessing_pipelines": ["default"],
    "shuffle_features": False,
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
                validation_protocol=ValidationProtocol.custom(num_bag_folds=2),
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


def _build_single(model_name, *, hyperparameters=None, **bundle_kwargs):
    entry = (model_name, 0) if hyperparameters is None else (model_name, 0, hyperparameters)
    bundle = BeyondArenaExperimentBundle(models=[entry], **bundle_kwargs)
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    assert len(experiments) == 1
    return experiments[0]


def test_model_verbosity_overrides_apply_per_model_and_yield_to_the_tuple():
    """CatBoost defaults to ag.verbosity 2, other models keep model_verbosity, the 3-tuple wins."""
    assert _model_hyperparameters(_build_single("CatBoost"))["ag.verbosity"] == 2
    assert _model_hyperparameters(_build_single("LightGBM"))["ag.verbosity"] == 4
    assert _model_hyperparameters(_build_single("CatBoost", hyperparameters={"ag.verbosity": 3}))["ag.verbosity"] == 3
    assert _model_hyperparameters(_build_single("CatBoost", model_verbosity_overrides={}))["ag.verbosity"] == 4
    # No model-level verbosity at all: the override does not resurrect the key.
    assert "ag.verbosity" not in _model_hyperparameters(_build_single("CatBoost", model_verbosity=None))


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
# Validation protocol: the bundle bakes one in only when asked; the arena context stamps it otherwise
# ===========================================================================


def _build(bundle: BeyondArenaExperimentBundle):
    experiments = bundle.build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    assert len(experiments) == 1
    return experiments[0]


def _build_single_autogluon(*, agexp_kwargs, **bundle_kwargs):
    from tabarena.benchmark.experiment import AGExperiment

    exp = _build(BeyondArenaExperimentBundle(models=[("AutoGluon_custom", agexp_kwargs)], **bundle_kwargs))
    assert isinstance(exp, AGExperiment)
    return exp


def test_bundle_leaves_the_protocol_to_the_context_by_default():
    """The official bundles carry no protocol: bagged, holdout and full-AutoGluon experiments are unstamped."""
    bagged = _build(BeyondArenaExperimentBundle(models=[("LightGBM", 0)]))
    holdout = _build(BeyondArenaExperimentBundle(models=[("LightGBM", 0)], holdout_experiments=True))
    predictor = _build_single_autogluon(
        agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}, "num_bag_folds": 8}},
    )
    assert bagged.validation_protocol is None
    assert holdout.validation_protocol is None
    assert predictor.validation_protocol is None
    # No bagging count is baked into the bagged fit kwargs; the full predictor's own counts pass through.
    assert "num_bag_folds" not in bagged.method_kwargs["fit_kwargs"]
    assert "adapt_num_bag_folds_to_n_classes" not in bagged.method_kwargs["fit_kwargs"]
    assert predictor.method_kwargs["fit_kwargs"]["num_bag_folds"] == 8


def test_bundle_protocol_reaches_bagged_holdout_and_autogluon_experiments():
    protocol = ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2)
    bagged = _build(BeyondArenaExperimentBundle(models=[("LightGBM", 0)], validation_protocol=protocol))
    holdout = _build(
        BeyondArenaExperimentBundle(models=[("LightGBM", 0)], holdout_experiments=True, validation_protocol=protocol)
    )
    predictor = _build_single_autogluon(
        agexp_kwargs={"fit_kwargs": {"hyperparameters": {"GBM": {}}}}, validation_protocol=protocol
    )
    assert bagged.validation_protocol == holdout.validation_protocol == predictor.validation_protocol == protocol


def test_bundle_protocol_is_not_baked_into_outer_experiments():
    outer = _build(
        BeyondArenaExperimentBundle(
            models=[("LightGBM", 0)],
            outer_experiments=True,
            validation_protocol=ValidationProtocol.custom(num_bag_folds=3),
        )
    )
    assert outer.validation_protocol is None


def test_bundle_protocol_sizes_the_seed_blocks():
    """With a protocol known at build time the fold-config-wise seed block follows it; otherwise it is 8."""
    bundle_kwargs = {"models": [("LightGBM", 2)], "n_random_configs": 2}
    default_seeds = [
        e.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["model_random_seed"]
        for e in BeyondArenaExperimentBundle(**bundle_kwargs).build_experiments(
            time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4
        )
    ]
    custom_seeds = [
        e.method_kwargs["model_hyperparameters"]["ag_args_ensemble"]["model_random_seed"]
        for e in BeyondArenaExperimentBundle(
            validation_protocol=ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2), **bundle_kwargs
        ).build_experiments(time_limit=60, num_cpus=1, num_gpus=0, memory_limit=4)
    ]
    assert default_seeds == [0, 8, 16]
    assert custom_seeds == [0, 6, 12]


def test_bundle_normalizes_a_protocol_dict():
    assert BeyondArenaExperimentBundle(validation_protocol={"num_bag_folds": 3}).validation_protocol == (
        ValidationProtocol.custom(num_bag_folds=3)
    )


def test_autogluon_experiment_protocol_override_is_respected():
    """An explicit per-entry ``validation_protocol`` wins over the bundle's."""
    exp = _build_single_autogluon(
        agexp_kwargs={
            "fit_kwargs": {"hyperparameters": {"GBM": {}}},
            "validation_protocol": TABARENA_V0PT1_VALIDATION_PROTOCOL,
        },
        validation_protocol=BEYONDARENA_VALIDATION_PROTOCOL,
    )
    assert exp.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL


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
