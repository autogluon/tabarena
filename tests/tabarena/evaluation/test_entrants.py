from __future__ import annotations

from itertools import pairwise

import pandas as pd
import pytest

from tabarena.evaluation.entrants import (
    DEFAULT_ENTRANT_POOL,
    ENTRANT_POOLS,
    filter_results_to_pool,
    get_entrant_pool,
)

# One representative entrant per (class, tags) combination the pools discriminate on.
_MODEL = ("model", ())
_OPEN_SYSTEM = ("system", ())
_LLM_SYSTEM = ("system", ("with-llm",))
_API_SYSTEM = ("system", ("closed-source-api",))
_API_LLM_SYSTEM = ("system", ("closed-source-api", "with-llm"))


@pytest.mark.parametrize(
    ("pool_key", "expected"),
    [
        # A model always competes; systems are admitted only where their tags are allowed.
        ("models", [_MODEL]),
        ("systems_open", [_MODEL, _OPEN_SYSTEM]),
        ("systems_llm", [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM]),
        ("systems_all", [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM, _API_SYSTEM, _API_LLM_SYSTEM]),
    ],
)
def test_pool_admission(pool_key, expected):
    pool = get_entrant_pool(pool_key)
    entrants = [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM, _API_SYSTEM, _API_LLM_SYSTEM]
    assert [e for e in entrants if pool.admits(*e)] == expected


def test_pools_are_cumulative():
    """Each pool admits everything the previous one does, so the selector reads as a ladder."""
    entrants = [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM, _API_SYSTEM, _API_LLM_SYSTEM]
    admitted = [{e for e in entrants if pool.admits(*e)} for pool in ENTRANT_POOLS]
    for narrower, wider in pairwise(admitted):
        assert narrower < wider


def test_default_pool_is_models_only():
    # The website opens on this one, so a change here changes the published front page.
    assert DEFAULT_ENTRANT_POOL.key == "models"
    assert not DEFAULT_ENTRANT_POOL.include_systems


def test_get_entrant_pool_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown entrant pool"):
        get_entrant_pool("systems_open_source")


def _info_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"method": "TabM", "suite": "s", "method_class": "model", "tags": ()},
            {"method": "AutoGluon", "suite": "s", "method_class": "system", "tags": ()},
            {"method": "Agent", "suite": "s", "method_class": "system", "tags": ("with-llm",)},
            {"method": "HostedAPI", "suite": "s", "method_class": "system", "tags": ("closed-source-api",)},
        ]
    )


def _results_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ta_name": ["TabM", "AutoGluon", "Agent", "HostedAPI"],
            "ta_suite": ["s", "s", "s", "s"],
        }
    )


@pytest.mark.parametrize(
    ("pool_key", "expected"),
    [
        ("models", ["TabM"]),
        ("systems_open", ["TabM", "AutoGluon"]),
        ("systems_llm", ["TabM", "AutoGluon", "Agent"]),
        ("systems_all", ["TabM", "AutoGluon", "Agent", "HostedAPI"]),
    ],
)
def test_filter_results_to_pool(pool_key, expected):
    kept = filter_results_to_pool(_results_frame(), get_entrant_pool(pool_key), _info_frame())
    assert list(kept["ta_name"]) == expected


def test_results_for_unknown_methods_are_kept():
    """A results row with no metadata entry defaults to being a model, which is what every
    result predating `method_class` is. Silently dropping it would corrupt the leaderboard.
    """
    df = pd.DataFrame({"ta_name": ["Mystery"], "ta_suite": ["s"]})
    kept = filter_results_to_pool(df, get_entrant_pool("models"), _info_frame())
    assert list(kept["ta_name"]) == ["Mystery"]


def test_info_frame_without_the_new_columns_keeps_everything():
    # Guards the path where an older collection has no method_class/tags columns at all.
    info = pd.DataFrame([{"method": "TabM", "suite": "s"}, {"method": "AutoGluon", "suite": "s"}])
    df = pd.DataFrame({"ta_name": ["TabM", "AutoGluon"], "ta_suite": ["s", "s"]})
    kept = filter_results_to_pool(df, get_entrant_pool("models"), info)
    assert list(kept["ta_name"]) == ["TabM", "AutoGluon"]
