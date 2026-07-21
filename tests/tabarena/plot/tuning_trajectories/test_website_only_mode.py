from __future__ import annotations

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
