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
    """Train time (the tuning budget) first, then the leaderboard's other time axes in the selector's order: the
    display columns alone, or every aggregated ``time_*`` column when the leaderboard has them.
    """
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(), fig_save_dir=tmp_path / "display", file_ext=".png", website_only=True
    )
    assert _explorer_x_axes(tmp_path / "display" / "tuning_trajectories_explorer.html") == [
        "x_train", "x_total", "x_infer", "x_total_task_mean", "x_infer_task_mean", "x_train_task_mean",
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
    assert axes[0] == "x_train" and len(axes) == 12


def test_the_explorer_title_is_passed_through(tmp_path):
    plot_tuning_trajectories_from_leaderboard(
        leaderboard=_toy_leaderboard(), fig_save_dir=tmp_path, file_ext=".png", website_only=True,
        explorer_title="classification",
    )  # fmt: skip
    html = (tmp_path / "tuning_trajectories_explorer.html").read_text(encoding="utf-8")
    assert json.loads(re.search(r"const CONFIG = (\{.*?\});\n", html, re.S).group(1))["title"] == "classification"
