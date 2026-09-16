"""Every hosted method declares the validation protocol its results were produced under."""

from __future__ import annotations

from tabarena.benchmark.validation_protocol import (
    BEYONDARENA_VALIDATION_PROTOCOL,
    TABARENA_V0PT1_VALIDATION_PROTOCOL,
)
from tabarena.contexts import BeyondArenaContext, TabArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_collection
from tabarena.contexts.tabarena.methods import (
    tabarena_method_metadata_2025_06_12_collection,
    tabarena_method_metadata_collection,
    tabarena_method_metadata_complete_collection,
)


def test_every_tabarena_method_declares_its_protocol():
    for collection in (
        tabarena_method_metadata_collection,
        tabarena_method_metadata_complete_collection,
        tabarena_method_metadata_2025_06_12_collection,
    ):
        for method in collection.method_metadata_lst:
            assert method.validation_protocol is not None, method.method
            if method.method_class == "system" and method.method_type != "portfolio":
                assert method.validation_protocol == "system", method.method
            else:
                assert method.validation_protocol == TABARENA_V0PT1_VALIDATION_PROTOCOL.key(), method.method


def test_every_beyondarena_method_declares_the_beyondarena_protocol():
    for method in beyond_method_metadata_collection.method_metadata_lst:
        assert method.validation_protocol == BEYONDARENA_VALIDATION_PROTOCOL.key(), method.method


def test_roster_methods_read_as_official_on_their_arena():
    tabarena = TabArenaContext(methods=[], task_metadata="tabarena")
    statuses = {tabarena.validation_protocol_status(m) for m in tabarena_method_metadata_collection.method_metadata_lst}
    assert statuses == {"official", "system"}
    beyond = BeyondArenaContext(methods=[], task_metadata="BeyondArena")
    assert {beyond.validation_protocol_status(m) for m in beyond_method_metadata_collection.method_metadata_lst} == {
        "official"
    }
    # A TabArena result read through the BeyondArena context is a custom protocol there, and the other way round.
    tabarena_model = next(
        m for m in tabarena_method_metadata_collection.method_metadata_lst if m.method_class == "model"
    )
    assert beyond.validation_protocol_status(tabarena_model) == "custom"
    assert tabarena.validation_protocol_status(beyond_method_metadata_collection.method_metadata_lst[0]) == "custom"
