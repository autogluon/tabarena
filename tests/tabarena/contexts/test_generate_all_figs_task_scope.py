"""`generate_all_figs` must hand the compare pass and the tuning-trajectory pass the same tasks.

A wider trajectory grid is not a cosmetic difference: the trajectory pass fills every task a
method does not cover and then drops any method carrying an imputed row, so the leaderboard can
report a method the trajectory figures silently omit.
"""

from __future__ import annotations

import pandas as pd
import pytest

from tabarena.contexts.abstract_arena_context import AbstractArenaContext

_resolve = AbstractArenaContext._resolve_shared_task_scope


def test_top_level_value_is_returned_and_dicts_untouched():
    compare_kwargs, trajectory_kwargs = {"plot": False}, {"file_ext": ".png"}
    assert _resolve("folds", [0, 1, 2], compare_kwargs, trajectory_kwargs) == [0, 1, 2]
    assert compare_kwargs == {"plot": False}
    assert trajectory_kwargs == {"file_ext": ".png"}


def test_absent_everywhere_is_none():
    assert _resolve("datasets", None, {}, {}) is None


@pytest.mark.parametrize("source", ["compare_kwargs", "tuning_trajectory_kwargs"])
def test_value_in_either_dict_is_hoisted_out(source):
    """Callers that passed the scope to one pass now get it applied to both, not just that one."""
    dicts = {"compare_kwargs": {}, "tuning_trajectory_kwargs": {}}
    dicts[source]["datasets"] = ["a", "b"]
    assert _resolve("datasets", None, **dicts) == ["a", "b"]
    # Popped, so the caller cannot also forward it positionally and duplicate the argument.
    assert dicts["compare_kwargs"] == {}
    assert dicts["tuning_trajectory_kwargs"] == {}


def test_agreeing_values_across_sources_collapse_to_one():
    assert _resolve("folds", [0], {"folds": [0]}, {"folds": [0]}) == [0]


def test_none_in_a_dict_does_not_override_a_real_value():
    assert _resolve("folds", [0, 1], {"folds": None}, {}) == [0, 1]


@pytest.mark.parametrize(
    ("top_level", "compare_kwargs", "trajectory_kwargs"),
    [
        ([0, 1], {"folds": [0, 1, 2]}, {}),
        (None, {"folds": [0]}, {"folds": [1]}),
        ([0], {}, {"folds": [0, 1]}),
    ],
)
def test_conflicting_values_raise(top_level, compare_kwargs, trajectory_kwargs):
    with pytest.raises(ValueError, match="Conflicting 'folds'"):
        _resolve("folds", top_level, compare_kwargs, trajectory_kwargs)


def test_generate_all_figs_forwards_the_same_scope_to_both_passes(monkeypatch, tmp_path):
    """End-to-end on the wiring: whichever dict the caller used, both passes see one grid."""
    seen: dict[str, dict] = {}

    def fake_compare(self, output_dir, **kwargs):
        seen["compare"] = kwargs
        return pd.DataFrame()

    def fake_plot(self, save_path, **kwargs):
        seen["trajectory"] = kwargs

    monkeypatch.setattr(AbstractArenaContext, "compare", fake_compare)
    monkeypatch.setattr(AbstractArenaContext, "plot_tuning_trajectories", fake_plot)

    AbstractArenaContext.generate_all_figs(
        object.__new__(AbstractArenaContext),
        output_dir=tmp_path,
        subsets=[[]],
        # Deliberately given the old way, on the compare side only.
        compare_kwargs={"folds": [0, 1, 2], "datasets": ["d1"]},
        plot_compare=True,
        plot_tuning_trajectories=True,
    )

    for pass_name in ("compare", "trajectory"):
        assert seen[pass_name]["folds"] == [0, 1, 2], pass_name
        assert seen[pass_name]["datasets"] == ["d1"], pass_name
