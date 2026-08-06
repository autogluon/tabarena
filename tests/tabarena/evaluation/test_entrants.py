from __future__ import annotations

import pandas as pd
import pytest

from tabarena.evaluation.entrants import (
    DEFAULT_ENTRANT_POOL,
    ENTRANT_POOLS,
    SYSTEM_CATEGORIES,
    filter_results_to_pool,
    get_entrant_pool,
    pool_key,
)
from tabarena.evaluation.eval_all import get_pool_reference_lines

# One representative entrant per (class, tags) combination the categories discriminate on.
_MODEL = ("model", ())
_OPEN_SYSTEM = ("system", ())
_LLM_SYSTEM = ("system", ("with-llm",))
_API_SYSTEM = ("system", ("closed-source-api",))
_API_LLM_SYSTEM = ("system", ("closed-source-api", "with-llm"))
_ALL = [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM, _API_SYSTEM, _API_LLM_SYSTEM]


@pytest.mark.parametrize(
    ("pool_key_", "expected"),
    [
        ("models", [_MODEL]),
        ("open", [_MODEL, _OPEN_SYSTEM]),
        # The point of independent toggles: LLM systems without the plain open-source ones.
        ("llm", [_MODEL, _LLM_SYSTEM]),
        ("api", [_MODEL, _API_SYSTEM]),
        ("open_llm", [_MODEL, _OPEN_SYSTEM, _LLM_SYSTEM]),
        ("open_api", [_MODEL, _OPEN_SYSTEM, _API_SYSTEM]),
        # A system carrying both tags needs both categories, so it appears only here and below.
        ("llm_api", [_MODEL, _LLM_SYSTEM, _API_SYSTEM, _API_LLM_SYSTEM]),
        ("open_llm_api", _ALL),
    ],
)
def test_pool_admission(pool_key_, expected):
    pool = get_entrant_pool(pool_key_)
    assert [e for e in _ALL if pool.admits(*e)] == expected


def test_every_combination_of_categories_is_published():
    assert len(ENTRANT_POOLS) == 2 ** len(SYSTEM_CATEGORIES)
    assert len({p.key for p in ENTRANT_POOLS}) == len(ENTRANT_POOLS)


def test_pool_key_is_order_independent():
    # The reader ticks boxes in any order; the folder segment must not depend on that.
    assert pool_key(["llm", "open"]) == pool_key(["open", "llm"]) == "open_llm"
    assert pool_key([]) == "models"


def test_a_multi_tagged_system_needs_every_one_of_its_categories():
    """Never admitted on the strength of a property the reader excluded."""
    assert not get_entrant_pool("llm").admits(*_API_LLM_SYSTEM)
    assert not get_entrant_pool("api").admits(*_API_LLM_SYSTEM)
    assert get_entrant_pool("llm_api").admits(*_API_LLM_SYSTEM)


def test_default_pool_is_models_only():
    # The website opens on this one, so a change here changes the published front page.
    assert DEFAULT_ENTRANT_POOL.key == "models"
    assert not DEFAULT_ENTRANT_POOL.categories


def test_get_entrant_pool_rejects_unknown_keys():
    with pytest.raises(ValueError, match="Unknown entrant pool"):
        get_entrant_pool("systems_all")


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
    ("pool_key_", "expected"),
    [
        ("models", ["TabM"]),
        ("open", ["TabM", "AutoGluon"]),
        ("llm", ["TabM", "Agent"]),
        ("api", ["TabM", "HostedAPI"]),
        ("open_llm_api", ["TabM", "AutoGluon", "Agent", "HostedAPI"]),
    ],
)
def test_filter_results_to_pool(pool_key_, expected):
    kept = filter_results_to_pool(_results_frame(), get_entrant_pool(pool_key_), _info_frame())
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


# -- reference lines / leaderboard admission ---------------------------------------------------
# `get_pool_reference_lines` is not only a plotting concern: `LeaderboardReporter.eval` keeps a
# non-config row only when its method is named in `baselines`, so a system left out here is
# deleted from the pool's published numbers.


def _named_info_frame() -> pd.DataFrame:
    info = _info_frame()
    info["display_name"] = info["method"]
    return info


@pytest.mark.parametrize(
    ("pool_key_", "expected"),
    [
        ("models", []),
        ("open", ["AutoGluon"]),
        ("llm", ["Agent"]),
        ("api", ["HostedAPI"]),
        ("open_llm_api", ["AutoGluon", "Agent", "HostedAPI"]),
    ],
)
def test_reference_lines_are_every_admitted_system(pool_key_, expected):
    names, colors = get_pool_reference_lines(pool_key_, _named_info_frame())
    assert names == expected
    assert len(colors) == len(names)


def test_reference_lines_never_list_a_model():
    names, _ = get_pool_reference_lines("open_llm_api", _named_info_frame())
    assert "TabM" not in names


def test_the_first_reference_colors_are_the_ones_autogluon_has_always_used():
    _, colors = get_pool_reference_lines("open_llm_api", _named_info_frame())
    assert colors[:2] == ["black", "tab:purple"]


def test_reference_lines_cover_every_system_in_the_shipped_collection():
    """The regression this guards: a system registered in the collection but absent from the
    widest pool's reference lines never reaches the leaderboard, silently.
    """
    from tabarena.contexts.tabarena.methods import tabarena_method_metadata_collection

    info = tabarena_method_metadata_collection.info()
    names, _ = get_pool_reference_lines("open_llm_api", info)
    systems = {m.display_name for m in tabarena_method_metadata_collection.method_metadata_lst if m.is_system}
    assert systems == set(names)
