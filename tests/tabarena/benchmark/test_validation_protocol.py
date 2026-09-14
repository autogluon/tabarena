"""`ValidationProtocol`: the official constants, resolution, identity, serialization and the compliance check."""

from __future__ import annotations

import pickle
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pytest

from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    OFFICIAL_VALIDATION_PROTOCOLS,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
    ValidationExpectation,
    ValidationProtocol,
    ValidationProtocolError,
    ValidationResolution,
    check_experiments,
    experiment_violations,
    validation_protocol_key,
)

# ---------------------------------------------------------------------------
# The official constants
# ---------------------------------------------------------------------------


def test_tabarena_protocol_is_eight_by_one_without_any_policy():
    p = TABARENA_V0PT1_VALIDATION_PROTOCOL
    assert (p.num_bag_folds, p.num_bag_sets) == (8, 1)
    assert not p.has_tiny_regime
    assert p.task_specific_validation is False
    assert p.adapt_num_folds_to_n_classes is False
    assert p.key() == "8x1"
    assert p.name == "TabArena-v0.1"


def test_beyondarena_protocol_is_eight_by_one_with_tiny_regime_and_task_awareness():
    p = BEYONDARENA_VALIDATION_PROTOCOL
    assert (p.num_bag_folds, p.num_bag_sets) == (8, 1)
    assert (p.tiny_num_bag_folds, p.tiny_num_bag_sets, p.tiny_max_group_instances) == (5, 5, 500)
    assert p.task_specific_validation is True
    assert p.adapt_num_folds_to_n_classes is True
    assert p.key() == "8x1+tiny5x5<=500+task-specific+adapt-classes"


def test_official_registry_holds_both_constants_by_name():
    assert set(OFFICIAL_VALIDATION_PROTOCOLS) == {"TabArena-v0.1", "BeyondArena"}
    assert all(p.is_official for p in OFFICIAL_VALIDATION_PROTOCOLS.values())


# ---------------------------------------------------------------------------
# Equality and identity ignore the labels
# ---------------------------------------------------------------------------


def test_default_protocol_equals_tabarena_constant_and_labels_do_not_count():
    assert ValidationProtocol() == TABARENA_V0PT1_VALIDATION_PROTOCOL
    relabelled = TABARENA_V0PT1_VALIDATION_PROTOCOL.with_origin(arena="TabArena", enforced=True)
    assert relabelled == TABARENA_V0PT1_VALIDATION_PROTOCOL
    assert relabelled.key() == "8x1"
    assert relabelled.arena == "TabArena"
    assert relabelled.enforced is True
    assert hash(relabelled) == hash(TABARENA_V0PT1_VALIDATION_PROTOCOL)


def test_a_custom_protocol_cannot_spoof_official_status_through_its_name():
    spoof = ValidationProtocol(num_bag_folds=2, name="TabArena-v0.1")
    assert spoof != TABARENA_V0PT1_VALIDATION_PROTOCOL
    assert spoof.key() == "2x1"
    assert spoof.is_official is False


def test_custom_helper_names_after_the_counts_and_accepts_fields():
    p = ValidationProtocol.custom(num_bag_folds=3, num_bag_sets=2)
    assert (p.num_bag_folds, p.num_bag_sets) == (3, 2)
    assert p.name == "custom-3x2"
    assert p.key() == "3x2"
    named = ValidationProtocol.custom(num_bag_folds=3, name="mine", task_specific_validation=True)
    assert named.name == "mine"
    assert named.key() == "3x1+task-specific"


def test_custom_helper_takes_keyword_arguments_only():
    with pytest.raises(TypeError):
        ValidationProtocol.custom(3, 2)  # type: ignore[misc]


def test_protocol_is_frozen():
    with pytest.raises(FrozenInstanceError):
        TABARENA_V0PT1_VALIDATION_PROTOCOL.num_bag_folds = 2  # type: ignore[misc]


# ---------------------------------------------------------------------------
# Validation of the fields
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_bag_folds": 1},
        {"num_bag_folds": True},
        {"num_bag_sets": 0},
        {"num_bag_folds": "eight"},
        {"tiny_num_bag_folds": 5},  # half-set regime
        {"tiny_num_bag_folds": 5, "tiny_num_bag_sets": 5},
        {"tiny_num_bag_folds": 1, "tiny_num_bag_sets": 5, "tiny_max_group_instances": 500},
        {"tiny_num_bag_folds": 5, "tiny_num_bag_sets": 0, "tiny_max_group_instances": 500},
        {"tiny_num_bag_folds": 5, "tiny_num_bag_sets": 5, "tiny_max_group_instances": -1},
    ],
)
def test_invalid_fields_are_rejected(kwargs):
    with pytest.raises(ValueError):
        ValidationProtocol(**kwargs)


# ---------------------------------------------------------------------------
# Resolution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("num_group_instances", [None, 50, 500, 501, 100_000])
def test_tabarena_resolves_to_eight_by_one_for_every_size(num_group_instances):
    assert TABARENA_V0PT1_VALIDATION_PROTOCOL.resolve_num_splits(num_group_instances) == (8, 1)
    assert TABARENA_V0PT1_VALIDATION_PROTOCOL.regime(num_group_instances) == "default"


@pytest.mark.parametrize(
    ("num_group_instances", "expected"),
    [(None, (8, 1)), (50, (5, 5)), (500, (5, 5)), (501, (8, 1)), (100_000, (8, 1))],
)
def test_beyondarena_resolves_tiny_regime_at_or_below_the_threshold(num_group_instances, expected):
    assert BEYONDARENA_VALIDATION_PROTOCOL.resolve_num_splits(num_group_instances) == expected


def test_tiny_regime_does_not_require_task_specific_validation():
    p = ValidationProtocol(tiny_num_bag_folds=5, tiny_num_bag_sets=5, tiny_max_group_instances=500)
    assert p.task_specific_validation is False
    assert p.resolve_num_splits(120) == (5, 5)
    assert p.key() == "8x1+tiny5x5<=500"


# ---------------------------------------------------------------------------
# Display
# ---------------------------------------------------------------------------


def test_describe_names_the_protocol_and_its_rules():
    assert TABARENA_V0PT1_VALIDATION_PROTOCOL.describe() == "TabArena-v0.1 (8 folds x 1 set)"
    assert BEYONDARENA_VALIDATION_PROTOCOL.describe() == (
        "BeyondArena (8 folds x 1 set; 5 x 5 at or below 500 training group instances; "
        "task-specific inner splits; class-adaptive folds)"
    )
    assert ValidationProtocol(num_bag_folds=3, num_bag_sets=2).describe() == "custom (3 folds x 2 sets)"


# ---------------------------------------------------------------------------
# (De)serialization
# ---------------------------------------------------------------------------


def test_to_dict_from_dict_round_trip_keeps_fields_and_labels():
    p = BEYONDARENA_VALIDATION_PROTOCOL.with_origin(arena="BeyondArena", enforced=False)
    data = p.to_dict()
    assert data["name"] == "BeyondArena"
    assert data["arena"] == "BeyondArena"
    assert data["enforced"] is False
    assert ValidationProtocol.from_dict(data) == p
    assert ValidationProtocol.from_dict(data).enforced is False


def test_from_dict_ignores_unknown_keys_and_from_config_normalizes():
    p = ValidationProtocol.from_dict({"num_bag_folds": 4, "future_field": 1})
    assert p.key() == "4x1"
    assert ValidationProtocol.from_config(None) is None
    assert ValidationProtocol.from_config(p) is p
    assert ValidationProtocol.from_config({"num_bag_folds": 4}) == p
    with pytest.raises(TypeError):
        ValidationProtocol.from_config(8)  # type: ignore[arg-type]


def test_stored_dict_with_an_official_name_but_other_fields_loads_without_error():
    p = ValidationProtocol.from_dict({"name": "BeyondArena", "num_bag_folds": 2})
    assert p.name == "BeyondArena"
    assert p.is_official is False


def test_pickle_round_trip_preserves_equality():
    p = BEYONDARENA_VALIDATION_PROTOCOL
    assert pickle.loads(pickle.dumps(p)) == p


# ---------------------------------------------------------------------------
# ValidationExpectation
# ---------------------------------------------------------------------------


def test_expectation_round_trip_and_official_flavours_fallback():
    e = ValidationExpectation(protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, enforced=True, arena="TabArena")
    data = e.to_dict()
    assert data["official_flavours"] == ["bagged", "system"]
    assert ValidationExpectation.from_dict(data) == e
    # An older sidecar without the flavours falls back to the default.
    legacy = {k: v for k, v in data.items() if k != "official_flavours"}
    assert ValidationExpectation.from_dict(legacy).official_flavours == ("bagged", "system")
    none = ValidationExpectation(protocol=None, enforced=False, arena="Arena")
    assert ValidationExpectation.from_dict(none.to_dict()) == none


# ---------------------------------------------------------------------------
# ValidationResolution
# ---------------------------------------------------------------------------


def test_resolution_record_is_a_plain_dict():
    r = ValidationResolution(
        num_group_instances=120,
        regime="tiny",
        num_bag_folds_nominal=5,
        num_bag_sets_nominal=5,
        num_bag_folds_resolved=3,
        num_bag_sets_resolved=1,
        clamps=("folds_capped_by_n_groups",),
        custom_splits=True,
        num_custom_splits=3,
        structure={"group_on": "grp"},
    )
    record = r.to_record()
    assert record["clamps"] == ["folds_capped_by_n_groups"]
    assert record["num_bag_folds_resolved"] == 3
    assert record["structure"] == {"group_on": "grp"}


# ---------------------------------------------------------------------------
# validation_protocol_key: the composite identity of a result record
# ---------------------------------------------------------------------------


def _record(flavour, protocol):
    return {"flavour": flavour, "protocol": None if protocol is None else protocol.to_dict()}


def test_key_of_a_missing_record_is_none():
    assert validation_protocol_key(None) is None
    assert validation_protocol_key({}) is None


def test_key_of_bagged_results_is_the_protocol_key():
    assert validation_protocol_key(_record("bagged", TABARENA_V0PT1_VALIDATION_PROTOCOL)) == "8x1"
    assert validation_protocol_key(_record("bagged", ValidationProtocol.custom(num_bag_folds=3))) == "3x1"
    assert validation_protocol_key(_record("bagged", None)) == "custom"


def test_key_of_non_bagged_results_never_reads_as_official():
    assert validation_protocol_key(_record("system", None)) == "system"
    assert validation_protocol_key(_record("system", TABARENA_V0PT1_VALIDATION_PROTOCOL)) == "system"
    assert validation_protocol_key(_record("holdout", TABARENA_V0PT1_VALIDATION_PROTOCOL)) == "holdout:8x1"
    assert validation_protocol_key(_record("outer", None)) == "outer"
    assert validation_protocol_key(_record("predictor", None)) == "predictor"
    assert validation_protocol_key(_record("bag-child-holdout", TABARENA_V0PT1_VALIDATION_PROTOCOL)) == (
        "bag-child-holdout:8x1"
    )
    assert validation_protocol_key(_record(None, None)) == "custom"
    assert validation_protocol_key(_record(None, TABARENA_V0PT1_VALIDATION_PROTOCOL)) == "custom:8x1"
    official_keys = {p.key() for p in OFFICIAL_VALIDATION_PROTOCOLS.values()}
    for flavour in ("holdout", "outer", "predictor", "bag-child-holdout", None):
        assert validation_protocol_key(_record(flavour, BEYONDARENA_VALIDATION_PROTOCOL)) not in official_keys


# ---------------------------------------------------------------------------
# experiment_violations / check_experiments
# ---------------------------------------------------------------------------


def _experiment(flavour, *, protocol=None, method_kwargs=None, name="exp"):
    return SimpleNamespace(
        VALIDATION_FLAVOUR=flavour, validation_protocol=protocol, method_kwargs=method_kwargs or {}, name=name
    )


def test_unstamped_or_matching_bagged_experiment_has_no_violation():
    protocol = TABARENA_V0PT1_VALIDATION_PROTOCOL
    assert experiment_violations(_experiment("bagged"), protocol=protocol, arena="TabArena") == []
    stamped = _experiment("bagged", protocol=protocol.with_origin(arena="TabArena", enforced=True))
    assert experiment_violations(stamped, protocol=protocol, arena="TabArena") == []


def test_a_foreign_stamp_is_a_violation_naming_both_protocols_and_the_pairing():
    violations = experiment_violations(
        _experiment("bagged", protocol=BEYONDARENA_VALIDATION_PROTOCOL, name="LightGBM_c1_BAG_L1"),
        protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL,
        arena="TabArena",
        bundle_hint="TabArenaV0pt1ExperimentBundle",
    )
    assert len(violations) == 1
    assert "LightGBM_c1_BAG_L1" in violations[0]
    assert "BeyondArena" in violations[0]
    assert "TabArena-v0.1" in violations[0]
    assert "TabArenaV0pt1ExperimentBundle" in violations[0]


def test_a_validation_metadata_override_is_a_violation():
    exp = _experiment("bagged", method_kwargs={"validation_metadata": {"group_on": None}})
    violations = experiment_violations(exp, protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, arena="TabArena")
    assert len(violations) == 1
    assert "validation_metadata" in violations[0]


@pytest.mark.parametrize("flavour", ["holdout", "outer", "system", "predictor", None])
def test_non_bagged_flavours_never_violate(flavour):
    exp = _experiment(
        flavour, protocol=ValidationProtocol.custom(num_bag_folds=2), method_kwargs={"validation_metadata": {}}
    )
    assert experiment_violations(exp, protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, arena="TabArena") == []


def test_objects_without_a_flavour_attribute_are_ignored():
    stub = SimpleNamespace(name="stub", model_constraints=None)
    assert experiment_violations(stub, protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, arena="TabArena") == []


def test_check_experiments_raises_one_error_listing_every_offender():
    expectation = ValidationExpectation(protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, enforced=True, arena="TabArena")
    good = _experiment("bagged", name="good")
    bad_a = _experiment("bagged", protocol=ValidationProtocol.custom(num_bag_folds=2), name="bad_a")
    bad_b = _experiment("bagged", method_kwargs={"validation_metadata": {"time_on": "t"}}, name="bad_b")
    with pytest.raises(ValidationProtocolError) as excinfo:
        check_experiments([good, bad_a, bad_b], expectation=expectation)
    message = str(excinfo.value)
    assert "bad_a" in message
    assert "bad_b" in message
    assert "good" not in message
    assert "official_validation_protocol=False" in message


def test_check_experiments_is_a_no_op_without_enforcement_or_protocol():
    bad = _experiment("bagged", protocol=ValidationProtocol.custom(num_bag_folds=2))
    check_experiments([bad], expectation=ValidationExpectation(protocol=None, enforced=True, arena="Arena"))
    check_experiments(
        [bad], expectation=ValidationExpectation(protocol=TABARENA_V0PT1_VALIDATION_PROTOCOL, enforced=False, arena="A")
    )
