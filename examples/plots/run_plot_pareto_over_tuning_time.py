"""Plot the predictive-performance vs. tuning-time Pareto trade-off.

``plot_tuning_trajectories()`` downloads TabArena's results (to
``~/.cache/tabarena/`` on first run) and draws, per method, how validation
performance improves as the tuning-time budget grows -- exposing the Pareto
frontier of accuracy against compute. With no arguments it writes figures to
``plots/n_configs``; pass ``fig_save_dir=...``, ``average_seeds=...``,
``file_ext=...`` etc. to customize (see the function signature for the full set
of options).
"""

from __future__ import annotations

from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories

if __name__ == "__main__":
    plot_tuning_trajectories()
