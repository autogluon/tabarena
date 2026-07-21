from __future__ import annotations

import matplotlib
import pandas as pd

matplotlib.use("Agg", force=True)

from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_hpo


def _toy_trajectories() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "name": ["A"] * 3 + ["B"] * 3 + ["C"] * 3,
            "time": [1.0, 10.0, 100.0, 2.0, 20.0, 200.0, 3.0, 30.0, 300.0],
            "imp": [10.0, 8.0, 7.0, 9.0, 6.0, 5.0, 20.0, 18.0, 17.0],
            "n_configs": [1, 4, 16] * 3,
        }
    )


def test_plot_hpo_focus_mode_writes_figure(tmp_path):
    save_path = tmp_path / "trajectories_focus.png"
    plot_hpo(
        df=_toy_trajectories(),
        xlabel="time",
        ylabel="imp",
        save_path=save_path,
        max_Y=False,
        method_col="name",
        sort_col="n_configs",
        focus_mode=True,
        family_map={"A": "Tree-based", "B": "Foundation Model", "C": "Neural Network"},
        focus_methods=["C"],
    )
    assert save_path.is_file()
    assert save_path.stat().st_size > 0


def test_plot_hpo_classic_mode_still_works(tmp_path):
    save_path = tmp_path / "trajectories_classic.png"
    plot_hpo(
        df=_toy_trajectories(),
        xlabel="time",
        ylabel="imp",
        save_path=save_path,
        max_Y=False,
        method_col="name",
        sort_col="n_configs",
    )
    assert save_path.is_file()
    assert save_path.stat().st_size > 0
