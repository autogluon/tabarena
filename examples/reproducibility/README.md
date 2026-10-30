## Reproducing the papers

These scripts regenerate the frozen results of the TabArena and BeyondArena papers, each pinned to
the method set the paper reports. Run them from this folder with `tabarena` installed as described in
the [repository README](../../README.md). Results download to `~/.cache/tabarena/` on first run,
and every script writes into an `output_*` folder here, which git ignores.

```bash
cd examples/reproducibility
python run_generate_paper_figures_beyondarena2026.py
```

### TabArena (NeurIPS 2025, [arXiv 2506.16791](https://arxiv.org/abs/2506.16791))

| Script | Output |
|---|---|
| `run_generate_main_leaderboard_neurips2025.py` | The paper's leaderboard (`output_leaderboard_neurips2025/`). |
| `run_generate_paper_figures_neurips2025.py` | All tables and figures of the paper (`output_paper_results/`). |

### BeyondArena (NeurIPS 2026, [arXiv 2606.30410](https://arxiv.org/abs/2606.30410))

The scripts evaluate `beyond_method_metadata_2026_camera_ready_collection`: the paper's 11 methods.

| Script | Output |
|---|---|
| `run_generate_main_leaderboard_beyondarena2026.py` | The leaderboard on every split, printed in the website's format (`output_leaderboard_beyondarena2026/`). |
| `run_generate_paper_figures_beyondarena2026.py` | The main-text figures and tables, the meta-feature regression and the runtime figure (`output_paper_results_beyondarena2026/`). |
| `run_generate_rank1_figure6_beyondarena2026.py` | Only the rank-1 share figure, which the figures script also writes (`output_figure6_beyondarena2026/`). |
| `run_generate_paper_per_dataset_figures_beyondarena2026.py` | The per-dataset appendix: a table and a tuning trajectory per dataset, and the LaTeX that sets them side by side (`output_paper_per_dataset_beyondarena2026/`). |
