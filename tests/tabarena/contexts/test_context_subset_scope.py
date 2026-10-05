"""`subset=` scopes a whole context to one slice of its tasks.

Within a context built with ``subset=S``, "all" is ``S`` and a subset ``T`` is ``S`` AND ``T``: the
tasks, and the results ``subset_results`` keeps, are those of the full context subset by ``[*S, *T]``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tabarena.contexts import BeyondArenaContext


def _tasks(ctx) -> set[tuple[str, int]]:
    grid = ctx.task_metadata_collection.task_grid()
    return set(zip(grid["dataset"], grid["split"].astype(int), strict=False))


def _results(ctx) -> pd.DataFrame:
    """One row per task of ``ctx`` for a dummy method (a results frame's ``fold`` is the split)."""
    grid = ctx.task_metadata_collection.task_grid()
    return pd.DataFrame(
        {
            "dataset": grid["dataset"].values,
            "fold": grid["split"].astype(int).values,
            "method": "m",
            "metric_error": np.arange(len(grid), dtype=float),
        }
    )


def _subset_of_full(full, subset: list[str]) -> set[tuple[str, int]]:
    kept = full.subset_results(_results(full), subset=subset)
    return set(zip(kept["dataset"], kept["fold"].astype(int), strict=False))


@pytest.fixture(scope="module")
def full():
    return BeyondArenaContext(task_metadata="BeyondArena", methods=[])


@pytest.fixture(scope="module")
def core2k():
    return BeyondArenaContext(task_metadata="BeyondArena", methods=[], subset="core2k")


def test_scope_keeps_exactly_the_subset_tasks(full, core2k):
    assert core2k.subset == ["core2k"]
    assert _tasks(core2k) == _subset_of_full(full, ["core2k"])
    assert len(_tasks(core2k)) < len(_tasks(full))


def test_scope_and_list_matches_compare_subsetting(full):
    ctx = BeyondArenaContext(task_metadata="BeyondArena", methods=[], subset=["core2k", "classification"])
    assert _tasks(ctx) == _subset_of_full(full, ["core2k", "classification"])


@pytest.mark.parametrize("subset", [["classification"], ["regression"], ["large"], ["!large"], ["tiny|small"]])
def test_subsets_within_the_scope_equal_the_full_context_and_list(full, core2k, subset):
    kept = core2k.subset_results(_results(core2k), subset=subset)
    assert set(zip(kept["dataset"], kept["fold"].astype(int), strict=False)) == _subset_of_full(
        full, ["core2k", *subset]
    )


def test_no_subset_keeps_every_task(full):
    assert full.subset == []
    assert _tasks(BeyondArenaContext(task_metadata="BeyondArena", methods=[], subset=[])) == _tasks(full)


def test_shortcut_name_expands_to_its_expressions(full):
    shortcuts = BeyondArenaContext.SUBSET_SHORTCUTS
    if not shortcuts:
        pytest.skip("BeyondArenaContext defines no subset shortcuts")
    name, expressions = next(iter(shortcuts.items()))
    ctx = BeyondArenaContext(task_metadata="BeyondArena", methods=[], subset=name)
    assert ctx.subset == list(expressions)
    assert _tasks(ctx) == _subset_of_full(full, list(expressions))


def test_empty_scope_raises():
    with pytest.raises(ValueError, match="matches no task"):
        BeyondArenaContext(task_metadata="BeyondArena", methods=[], subset=["core2k", "!core2k"])
