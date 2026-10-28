"""Reproduce the result figures and tables of the BeyondArena paper (arXiv 2606.30410).

Evaluates the paper's 11 methods (``beyond_method_metadata_2026_06_paper_collection``) on every
split of the 12 sub-benchmarks and writes, under ``save_path``:

* ``subsets/<subset>/``: each sub-benchmark's leaderboard. ``subsets/full/`` also holds the
  figures for all datasets: ``tuning-impact-elo.pdf`` (paper: ``full_tuning-impact-elo.pdf``),
  ``pareto_front_improvability_vs_time_infer.pdf`` and the LaTeX leaderboard ``leaderboard.tex``.
* ``result_plots/``: Elo and improvability per sub-benchmark, per model family
  (``per_family_elo.pdf``, paper: ``result_plot_per_family_plot.pdf``) and per model
  (``per_model_elo.pdf``, paper: ``result_plot_per_model.pdf``).
* ``tuning_trajectories/pareto_n_configs_imp.pdf``: improvability vs. train time over the
  number of tuned configurations.
* ``runtimes/``: the training-runtime distribution across configurations
  (paper: ``data-analysis-runtime-per-method.pdf``).

Results download to ``~/.cache/tabarena/`` on first run.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_2026_06_paper_collection
from tabarena.plot.subset_results import plot_subset_results

#: The paper's sub-benchmarks: label -> subset expressions, each on every split.
PAPER_SUBSETS: dict[str, list[str]] = {
    "random": ["random"],
    "grouped": ["grouped"],
    "temporal": ["temporal"],
    "tiny": ["tiny"],
    "small": ["small"],
    "medium": ["medium"],
    "large": ["large"],
    "low-dim": ["low-dim"],
    "high-dim": ["high-dim"],
    "text": ["text"],
    "high-cardinality": ["high-cardinality"],
    "full": ["all"],
}

if __name__ == "__main__":
    save_path = Path("output_paper_results_beyondarena2026")  # folder to save all figures and tables

    context = BeyondArenaContext(
        methods=beyond_method_metadata_2026_06_paper_collection.method_metadata_lst,
    )

    # The overview figures group methods by display name (e.g. "RealMLP", not "TA-REALMLP").
    rename = context.get_method_rename_map()
    leaderboards = {}
    for label, subset in PAPER_SUBSETS.items():
        leaderboard, results = context.compare(
            output_dir=save_path / "subsets" / label,
            subset=subset,
            return_results=True,
            plot=label == "full",  # the paper shows figures for all datasets only
        )
        leaderboard["method"] = leaderboard["config_type"].map(rename).fillna(leaderboard["method"])
        leaderboard["n_datasets_total"] = results["dataset"].nunique()  # the "N=" under each sub-benchmark
        leaderboards[label] = leaderboard
    plot_subset_results(leaderboards, save_path / "result_plots")

    context.plot_tuning_trajectories(
        save_path=save_path / "tuning_trajectories",
        exclude_imputed=False,
        # Color methods by each figure's own y-axis (improvability in pareto_n_configs_imp.pdf).
        use_elo_method_order=False,
        plot_kwargs={"ylim_imp": (None, None), "legend_in_plot": True, "figsize": (6, 4.5)},
    )

    context.plot_runtime_per_method(
        save_path=save_path / "runtimes",
        deep_dive_kwargs={"time_limit_soft": 10800, "time_limit_hard": 14400},  # the paper's 4 h limit
    )
