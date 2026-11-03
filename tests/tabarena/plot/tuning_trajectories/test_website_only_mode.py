from __future__ import annotations

import json
import re

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import (
    plot_tuning_trajectories_from_leaderboard,
)


def _toy_leaderboard() -> pd.DataFrame:
    rows = []
    for method, base_imp, base_elo in [("A", 12.0, 1300.0), ("B", 9.0, 1450.0)]:
        for i, n_configs in enumerate([1, 4, 16]):
            train_1k = 2.0 * (4**i)
            infer_1k = 0.1 * (2**i)
            imp = base_imp - i
            elo = base_elo + 20 * i
            rows.append(
                {
                    "name": method,
                    "n_configs": n_configs,
                    "Train time per 1K samples (s) (median)": train_1k,
                    "Inference time per 1K samples (s) (median)": infer_1k,
                    "Total time per 1K samples (s) (median)": train_1k + infer_1k,
                    "Train time (s)": train_1k * 10,
                    "Infer time (s)": infer_1k * 10,
                    "Total time (s)": (train_1k + infer_1k) * 10,
                    "Metric Error": imp / 100,
                    "Improvability (%)": imp,
                    "Improvability (%) (Test)": imp,
                    "Improvability (%) (Val)": imp - 1,
                    "Improvability (%) (Test) - Improvability (%) (Val)": 1.0,
                    "Elo": elo,
                    "Elo (Test)": elo,
                    "Elo (Val)": elo + 15,
                    "Elo (Val) - Elo (Test)": 15.0,
                    "Elo (Test) - Elo (Val)": -15.0,
                    "Baseline Advantage (%)": 5.0 + i,
                    "Baseline Advantage (%) (Test)": 5.0 + i,
                    "Baseline Advantage (%) (Val)": 6.0 + i,
                    "Baseline Advantage (%) (Test - Val)": -1.0,
                }
            )
    return pd.DataFrame(rows)


def test_website_only_renders_shipped_outputs_only(tmp_path):
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(),
        fig_save_dir=tmp_path,
        file_ext=".png",
        website_only=True,
    )
    # The shipped figure + interactive artifacts exist ...
    assert (tmp_path / "pareto_n_configs_imp.png").is_file()
    assert (tmp_path / "tuning_trajectories_explorer.html").is_file()
    assert (tmp_path / "tuning_trajectories.csv").is_file()
    # ... and the rest of the figure suite was skipped.
    assert not (tmp_path / "pareto_n_configs_elo.png").exists()
    assert not (tmp_path / "pareto_n_configs_imp_tot_train.png").exists()


def test_full_mode_renders_more_figures(tmp_path):
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(),
        fig_save_dir=tmp_path,
        file_ext=".png",
    )
    assert (tmp_path / "pareto_n_configs_imp.png").is_file()
    assert (tmp_path / "pareto_n_configs_elo.png").is_file()
    assert (tmp_path / "pareto_n_configs_imp_tot_train.png").is_file()


def test_uncovered_static_figures_leave_only_the_explorers(tmp_path):
    """Every static trajectory figure is a view of the trajectories or diagnostics explorer, so "uncovered" writes the
    two pages and their data alone.
    """
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(),
        fig_save_dir=tmp_path,
        file_ext=".png",
        static_figures="uncovered",
    )
    assert sorted(p.name for p in tmp_path.iterdir()) == [
        "tuning_diagnostics.csv",
        "tuning_diagnostics_explorer.html",
        "tuning_trajectories.csv",
        "tuning_trajectories_explorer.html",
    ]


def test_the_diagnostics_explorer_offers_every_diagnostic_view(tmp_path):
    """Metric error, Elo on validation, baseline advantage and the three test - validation gaps against time, and the
    validation and test metrics on linear x-axes; a method without validation results keeps its other views.
    """
    leaderboard = _toy_leaderboard()
    no_val = leaderboard["name"] == "B"
    for col in ("Elo (Val)", "Improvability (%) (Val)", "Baseline Advantage (%) (Val)", "Elo (Test) - Elo (Val)"):
        leaderboard.loc[no_val, col] = float("nan")
    plot_tuning_trajectories_from_leaderboard(leaderboard=leaderboard, fig_save_dir=tmp_path, file_ext=".png")
    html = (tmp_path / "tuning_diagnostics_explorer.html").read_text()
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html).group(1))
    assert [m["key"] for m in config["metrics"]] == [
        "err",
        "elo_val",
        "adv",
        "adv_gap",
        "elo_gap",
        "imp_gap",
        "imp",
        "elo",
    ]
    axes = {a["key"]: a.get("scale", "log") for a in config["xAxes"]}
    assert {k: v for k, v in axes.items() if v == "linear"} == dict.fromkeys(
        ["x_elo_val", "x_imp_val", "x_adv_val", "x_elo", "x_adv"], "linear"
    )
    assert [a["diagonalWith"] for a in config["xAxes"] if a["key"] == "x_elo_val"] == ["elo"]
    points = json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))
    assert {p["method"] for p in points} == {"A", "B"}
    assert all(p["elo_val"] is None for p in points if p["method"] == "B")


def test_a_shaded_system_carries_its_shade_into_the_explorer(tmp_path):
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(),
        fig_save_dir=tmp_path,
        file_ext=".png",
        focus_mode=True,
        website_only=True,
        system_names=frozenset({"B"}),
        system_shades={"B": "closed-source-api"},
    )
    html = (tmp_path / "tuning_trajectories_explorer.html").read_text(encoding="utf-8")
    points = json.loads(re.search(r"const POINTS = (\[.*?\]);\n", html, re.S).group(1))
    assert {p["method"]: p["shade"] for p in points} == {"A": None, "B": "closed-source-api"}


def _explorer_x_axes(path) -> list[str]:
    html = path.read_text(encoding="utf-8")
    config = json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))
    return [axis["key"] for axis in config["xAxes"]]


def test_the_explorer_offers_the_time_axes_the_leaderboard_carries(tmp_path):
    """The time axes the leaderboard carries in the Pareto explorer's order, opening on total time per 1K samples:
    the display columns alone, or every aggregated ``time_*`` column when the leaderboard has them.
    """
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(), fig_save_dir=tmp_path / "display", file_ext=".png", website_only=True
    )
    assert _explorer_x_axes(tmp_path / "display" / "tuning_trajectories_explorer.html") == [
        "x_total", "x_infer", "x_train", "x_total_task_mean", "x_infer_task_mean", "x_train_task_mean",
    ]  # fmt: skip
    leaderboard = _toy_leaderboard()
    for kind in ("train", "infer", "total"):
        for prefix in ("", "median_"):
            for suffix in ("", "_per_1K"):
                leaderboard[f"{prefix}time_{kind}_s{suffix}"] = leaderboard["Total time (s)"] + len(prefix + suffix)
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=leaderboard, fig_save_dir=tmp_path / "all", file_ext=".png", website_only=True
    )
    axes = _explorer_x_axes(tmp_path / "all" / "tuning_trajectories_explorer.html")
    assert axes[0] == "x_total" and len(axes) == 12


def test_the_explorer_title_is_passed_through(tmp_path):
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(), fig_save_dir=tmp_path, file_ext=".png", website_only=True,
        explorer_title="classification",
    )  # fmt: skip
    html = (tmp_path / "tuning_trajectories_explorer.html").read_text(encoding="utf-8")
    assert json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))["title"] == "classification"
