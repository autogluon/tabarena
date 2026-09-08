"""How ``generate_all_figs`` decides how many processes to run the subset passes in."""

from __future__ import annotations

import inspect

import pytest

from tabarena.contexts import abstract_arena_context
from tabarena.contexts.abstract_arena_context import AbstractArenaContext, _default_max_workers


@pytest.mark.parametrize(("available", "expected"), [(1, 1), (2, 1), (8, 7), (48, 47)])
def test_one_process_per_cpu_less_one(monkeypatch, available: int, expected: int):
    """A core is left for everything else, and a single-CPU machine still gets one worker."""
    monkeypatch.setattr(abstract_arena_context.os, "sched_getaffinity", lambda _pid: set(range(available)))

    assert _default_max_workers() == expected


def test_cpus_are_counted_as_the_ones_this_process_may_use(monkeypatch):
    """An affinity mask or container quota makes that smaller than the machine's CPU count."""
    monkeypatch.setattr(abstract_arena_context.os, "sched_getaffinity", lambda _pid: {0, 1, 2, 3})
    monkeypatch.setattr(abstract_arena_context.os, "cpu_count", lambda: 256)

    assert _default_max_workers() == 3


def test_falls_back_to_cpu_count_without_affinity_support(monkeypatch):
    """`sched_getaffinity` is Linux-only."""
    monkeypatch.delattr(abstract_arena_context.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(abstract_arena_context.os, "cpu_count", lambda: 4)

    assert _default_max_workers() == 3


def test_an_unknown_cpu_count_still_yields_a_worker(monkeypatch):
    monkeypatch.delattr(abstract_arena_context.os, "sched_getaffinity", raising=False)
    monkeypatch.setattr(abstract_arena_context.os, "cpu_count", lambda: None)

    assert _default_max_workers() == 1


def test_max_workers_defaults_to_auto():
    """`None` is the sentinel the auto path keys off; an explicit count must still be honoured."""
    assert inspect.signature(AbstractArenaContext.generate_all_figs).parameters["max_workers"].default is None


def test_the_composite_leaderboard_is_written_by_default():
    """`None` means "whenever the leaderboards it aggregates are produced"."""
    default = inspect.signature(AbstractArenaContext.generate_all_figs).parameters["save_composite_leaderboard"].default
    assert default is None


def test_asking_for_the_composite_without_the_compare_pass_is_an_error():
    """Explicitly requesting it with nothing to aggregate is a mistake worth naming."""
    with pytest.raises(ValueError, match="requires plot_compare=True"):
        AbstractArenaContext.generate_all_figs(
            AbstractArenaContext.__new__(AbstractArenaContext),
            output_dir="unused",
            plot_compare=False,
            save_composite_leaderboard=True,
        )


def test_a_trajectories_only_run_does_not_trip_over_the_default(monkeypatch):
    """The default must not turn `plot_compare=False` into a failure it never asked for."""
    captured = {}

    def fake_subset_figs(self, subset, **kwargs):
        captured["collect_composite"] = kwargs["collect_composite"]
        return ("", None)

    monkeypatch.setattr(AbstractArenaContext, "_generate_subset_figs", fake_subset_figs)
    monkeypatch.setattr(AbstractArenaContext, "_default_subsets", property(lambda self: [[]]))

    AbstractArenaContext.generate_all_figs(
        AbstractArenaContext.__new__(AbstractArenaContext), output_dir="unused", plot_compare=False
    )

    assert captured["collect_composite"] is False
