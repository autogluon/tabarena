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


def test_the_composite_pareto_explorer_holds_every_subset_with_points(tmp_path):
    """One page over the subsets' `pareto_front_points.csv`, in the subsets' order; a subset without points is left out."""
    import json
    import re
    from types import SimpleNamespace

    import pandas as pd

    for suffix, elo in (("all", 1500.0), ("tiny", 1400.0)):
        (tmp_path / suffix).mkdir()
        pd.DataFrame(
            {
                "method": ["A", "B"],
                "variant": ["Default", "Default"],
                "family": ["Tree-based", "Foundation Model"],
                "shade": [None, None],
                "x_infer": [0.1, 1.0],
                "x_train": [1.0, 5.0],
                "imp": [10.0, 5.0],
                "elo": [elo, elo + 100],
                "imputed": [False, False],
                "imputed_pct": [0.0, 0.0],
            }
        ).to_csv(tmp_path / suffix / "pareto_front_points.csv", index=False)
    (tmp_path / "large").mkdir()  # its compare pass wrote no points

    page = AbstractArenaContext._save_composite_pareto_explorer(
        SimpleNamespace(benchmark_name="Arena"), tmp_path, ["all", "large", "tiny"]
    )

    assert page == tmp_path / "pareto_front_explorer.html"
    html = page.read_text(encoding="utf-8")
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    assert [s["key"] for s in config["subsets"]] == ["all", "tiny"]
    assert '"subset":"tiny"' in html and 'id="subset-select"' in html
    assert "<title>Arena Pareto explorer</title>" in html


def test_the_composite_pareto_explorer_is_written_by_default():
    default = inspect.signature(AbstractArenaContext.generate_all_figs).parameters["save_composite_pareto_explorer"]
    assert default.default is None


def test_the_composite_trajectories_explorer_holds_every_subset(tmp_path):
    """One trajectories page over the subsets' `tuning_trajectories.csv`, in the subsets' order; a subset without
    trajectories is left out.
    """
    import json
    import re
    from types import SimpleNamespace

    import pandas as pd

    for suffix, elo in (("all", 1500.0), ("tiny", 1400.0)):
        (tmp_path / suffix / "tuning_trajectories").mkdir(parents=True)
        pd.DataFrame(
            {
                "method": ["A", "A", "B"],
                "family": ["Tree-based", "Tree-based", "Foundation Model"],
                "shade": [None, None, None],
                "x_train": [1.0, 20.0, 5.0],
                "imp": [10.0, 8.0, 5.0],
                "elo": [elo, elo + 50, elo + 100],
                "n_configs": [1, 8, 1],
            }
        ).to_csv(tmp_path / suffix / "tuning_trajectories" / "tuning_trajectories.csv", index=False)
    (tmp_path / "large").mkdir()  # no trajectories there

    page = AbstractArenaContext._save_composite_trajectories_explorer(
        SimpleNamespace(benchmark_name="Arena"), tmp_path, ["all", "large", "tiny"]
    )

    assert page == tmp_path / "tuning_trajectories_explorer.html"
    html = page.read_text(encoding="utf-8")
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    assert [s["key"] for s in config["subsets"]] == ["all", "tiny"]
    assert config["mode"] == "trajectory"
    assert "<title>Arena tuning trajectories</title>" in html


def test_the_composite_trajectories_explorer_follows_the_trajectories():
    default = inspect.signature(AbstractArenaContext.generate_all_figs).parameters[
        "save_composite_trajectories_explorer"
    ]
    assert default.default is None


def test_static_figures_reach_the_compare_and_trajectory_passes(monkeypatch):
    """By default the static figures an explorer covers are left out; explicit pass kwargs win."""
    captured = []

    def fake_subset_figs(self, subset, **kwargs):
        captured.append(kwargs)
        return ("", None)

    monkeypatch.setattr(AbstractArenaContext, "_generate_subset_figs", fake_subset_figs)
    monkeypatch.setattr(AbstractArenaContext, "_default_subsets", property(lambda self: [[]]))
    context = AbstractArenaContext.__new__(AbstractArenaContext)
    monkeypatch.setattr(AbstractArenaContext, "_save_composite_pareto_explorer", lambda *args: None)

    AbstractArenaContext.generate_all_figs(context, output_dir="unused", max_workers=1)
    assert captured[-1]["compare_kwargs"]["static_figures"] == "uncovered"
    assert captured[-1]["tuning_trajectory_kwargs"]["static_figures"] == "uncovered"

    AbstractArenaContext.generate_all_figs(
        context, output_dir="unused", max_workers=1, compare_kwargs={"static_figures": "all"}
    )
    assert captured[-1]["compare_kwargs"]["static_figures"] == "all"
    assert captured[-1]["tuning_trajectory_kwargs"]["static_figures"] == "uncovered"

    with pytest.raises(ValueError, match="static_figures"):
        AbstractArenaContext.generate_all_figs(context, output_dir="unused", max_workers=1, static_figures="none")
