"""``compare(exclude_systems=True)`` scores the new methods against individual models only."""

from __future__ import annotations

import copy

import pandas as pd
import pytest

import tabarena.nips2025_utils.compare as compare_module
from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_tabpfn35_metadata
from tabarena.systems.tabpfn_3_5_thinking.info import tabpfn_3_5_thinking_max_metadata


@pytest.fixture
def ctx() -> BeyondArenaContext:
    return BeyondArenaContext(
        task_metadata="BeyondArena", methods=[beyond_tabpfn35_metadata, tabpfn_3_5_thinking_max_metadata]
    )


def _new_system():
    system = copy.copy(tabpfn_3_5_thinking_max_metadata)
    system.method = "New-System"
    return system


def _scored_methods(ctx, monkeypatch, **compare_kwargs) -> set[str]:
    """The ``ta_name`` values ``compare`` hands to the scorer, which is stubbed out."""
    seen = {}

    def fake_compare(df_results, **kwargs):
        seen["df"] = df_results
        return pd.DataFrame({"method": sorted(df_results["method"].unique())})

    monkeypatch.setattr(compare_module, "compare", fake_compare)
    names = [m.method for m in ctx.method_metadata_collection.method_metadata_lst]
    ta_results = pd.DataFrame(
        {"dataset": "d", "fold": 0, "method": names, "ta_name": names, "metric_error": range(len(names))}
    )
    ctx.compare(output_dir=None, ta_results=ta_results, filter_to_task_metadata=False, **compare_kwargs)
    return set(seen["df"]["ta_name"])


def test_systems_lists_the_roster_systems_only(ctx):
    assert ctx.systems == ["TabPFN-3.5-Thinking_max"]


def test_exclude_systems_scores_the_models_only(ctx, monkeypatch):
    assert _scored_methods(ctx, monkeypatch) == {"TA-TabPFN-3.5", "TabPFN-3.5-Thinking_max"}
    assert _scored_methods(ctx, monkeypatch, exclude_systems=True) == {"TA-TabPFN-3.5"}


def test_a_system_registered_as_a_new_method_stays_in(ctx, monkeypatch):
    ctx.register([_new_system()], scope_to_valid_tasks=False)
    assert ctx.systems == ["TabPFN-3.5-Thinking_max"]
    assert _scored_methods(ctx, monkeypatch, exclude_systems=True) == {"TA-TabPFN-3.5", "New-System"}
