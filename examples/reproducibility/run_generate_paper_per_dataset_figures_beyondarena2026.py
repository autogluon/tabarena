r"""Reproduce the per-dataset appendix of the BeyondArena paper (arXiv 2606.30410).

For each of the 142 datasets, a table of every method's test error (default, tuned, tuned +
ensembled; mean and standard deviation over all splits) next to its HPO Pareto trajectory. Writes,
under ``save_path``, the same layout as the paper's ``paper/tables/per_dataset/`` folder:

* ``per-dataset-tables/fragments/<dataset>.tex``: the table of each dataset;
* ``per-dataset/<dataset>/tuning_trajectories/pareto_n_configs_err_tot_train.pdf``: its trajectory;
* ``per-dataset-combined/<dataset>.tex``: both side by side, and ``all_per_dataset.tex``, which
  inputs every dataset (the paper's ``\input{paper/tables/per_dataset/per-dataset-combined/all_per_dataset}``).

The LaTeX paths are rooted at ``tex_path_root``, so copy ``save_path``'s contents to that folder of the
paper. Results download to ``~/.cache/tabarena/`` on first run.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.contexts.beyondarena.methods import beyond_method_metadata_2026_06_paper_collection
from tabarena.nips2025_utils.per_dataset_tables import write_per_dataset_combined_tex

if __name__ == "__main__":
    save_path = Path("output_paper_per_dataset_beyondarena2026")  # folder to save all figures and tables
    tex_path_root = "paper/tables/per_dataset"  # where the paper's LaTeX finds the contents of save_path

    methods = beyond_method_metadata_2026_06_paper_collection.method_metadata_lst
    context = BeyondArenaContext(methods=methods)

    # Table rows by display name: every default, then every tuned, then every tuned + ensembled.
    display_names = [m.display_name for m in methods]
    context.generate_per_dataset_tables(
        save_path=save_path / "per-dataset-tables",
        per_dataset_dir=save_path / "per-dataset-tables" / "fragments",
        method_order=[
            f"{n} {regime}" for regime in ("(default)", "(tuned)", "(tuned + ensemble)") for n in display_names
        ],
        use_display_names=True,
    )

    context.plot_tuning_trajectories_per_dataset(
        save_path=save_path / "per-dataset",
        file_ext=".pdf",
        plot_kwargs={
            # Legend in table order; reversed colors give the last entry (TabICLv2) the first, darkest color.
            "method_order": [n for n in display_names if n != "Linear"],
            "reverse_colors": True,
            # Clip long "Dataset: <name>" titles so every figure keeps the same size.
            "clip_title": True,
            "max_title_chars": 40,
        },
    )

    write_per_dataset_combined_tex(
        output_dir=save_path,
        tex_path_root=tex_path_root,
        task_metadata=context.task_metadata_collection,
    )
