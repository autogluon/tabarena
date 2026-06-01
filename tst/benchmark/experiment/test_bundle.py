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
    "model_artifacts_base_path",
    "verbosity",
    "model_verbosity",
    "adapt_num_folds_to_n_classes",
    "shuffle_features",
    "dynamic_tabarena_validation_protocol",
    "custom_model_constraints",
}

# The defaults shared by every subclass (i.e. inherited from the parent and not
# overridden). Subclass-specific expectations below extend this baseline.
COMMON_INHERITED_DEFAULTS = {
    "models": [],
    "model_agnostic_preprocessing": True,
    "max_predict_batch_size": None,
    "sequential_local_fold_fitting": False,
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
}

TABARENA_V0PT1_EXPECTED = {
    **COMMON_INHERITED_DEFAULTS,
    "n_random_configs": 200,
    "preprocessing_pipelines": ["default"],
    "adapt_num_folds_to_n_classes": False,
    "shuffle_features": False,
    "dynamic_tabarena_validation_protocol": False,
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
