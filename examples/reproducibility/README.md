## Reproducing the papers

These scripts regenerate the frozen results of the TabArena and BeyondArena papers, each pinned to
the method set the paper reports. Run them from this folder with `tabarena` installed as described in
the [repository README](../../README.md). Results download to `~/.cache/tabarena/` on first run,
and every script writes into an `output_*` folder here, which git ignores.

```bash
cd examples/reproducibility
python run_generate_paper_figures_beyondarena2026.py
```

### TabArena (NeurIPS 2025)

| Script | Output |
|---|---|
| `run_generate_main_leaderboard_neurips2025.py` | The paper's leaderboard (`output_leaderboard_neurips2025/`). |
| `run_generate_paper_figures_neurips2025.py` | All tables and figures of the paper (`output_paper_results/`). |

### BeyondArena ([arXiv 2606.30410](https://arxiv.org/abs/2606.30410))

The scripts evaluate `beyond_method_metadata_2026_camera_ready_collection`: the paper's 11 methods
with TabM's results after its constant-column and clipping fixes. Swap in
`beyond_method_metadata_2026_06_paper_collection` to get the numbers of arXiv v1.

| Script | Output |
|---|---|
| `run_generate_main_leaderboard_beyondarena2026.py` | The leaderboard on every split, printed in the website's format (`output_leaderboard_beyondarena2026/`). |
| `run_generate_paper_figures_beyondarena2026.py` | The main-text figures and tables, the meta-feature regression and the runtime figure (`output_paper_results_beyondarena2026/`). |
| `run_generate_paper_per_dataset_figures_beyondarena2026.py` | The per-dataset appendix: a table and a tuning trajectory per dataset, and the LaTeX that sets them side by side (`output_paper_per_dataset_beyondarena2026/`). |

Where each file of the paper comes from, relative to `output_paper_results_beyondarena2026/`:

| Paper file | Script output |
|---|---|
| `full_tuning-impact-elo.pdf` | `subsets/full/tuning-impact-elo.pdf` |
| `full_pareto_front_improvability_vs_time_infer.pdf` | `subsets/full/pareto_front_improvability_vs_time_infer.pdf` |
| `tables/all/leaderboard.tex` | `subsets/full/leaderboard.tex` |
| `result_plot_per_family_plot.pdf` | `result_plots/per_family_elo.pdf` |
| `result_plot_per_model.pdf` | `result_plots/per_model_elo.pdf` |
| `pareto_n_configs_imp.pdf` | `tuning_trajectories/pareto_n_configs_imp.pdf` |
| `per_family_elo_le100k.pdf` | `result_plots_le100k/per_family_elo.pdf` |
| `tuning-impact-elo-horizontal_tabpfn_3_large.pdf` | `subsets_large_with_tabpfn3/tuning-impact-elo-horizontal.pdf` |
| `data-analysis-runtime-per-method.pdf` | `runtimes/figures/data-analysis-runtime-per-method.pdf` |
| The meta-feature regression in the text | `metafeature_regression/coefficients.csv` |
| `tables/per_dataset/` | The contents of `output_paper_per_dataset_beyondarena2026/` |

These scripts do not regenerate `ta_vs_non_ta_elo_parity_distinct.pdf`, `rank_1_plot_updated.pdf`,
the dataset curation dashboard, the ablations or the statistical tests. The paper keeps them as
published.
