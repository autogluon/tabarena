"""Collect and preprocess data for leaderboard website.

Only the interactive artifacts ship. Every TabArena figure has a self-contained HTML explorer
that the site renders by default and that exports its own SVG / PDF / PNG on demand, so the
static PNGs were a second copy of what the reader already had: 80 of the 105 MB in the Space's
``data/``, and the only reason it needed Git LFS. They are still rendered into
``raw_website_artifacts/`` for paper use; they are simply not published.

BeyondArena is unaffected. It has its own copy path in
``scripts/run_generate_beyondarena_website_artifacts.py`` and ships PNGs because it has no
explorers for most of its figures.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pandas as pd

from tabarena.plot.interactive.leaderboard_explorer import build_leaderboard_explorer_html
from tabarena.plot.interactive.leaderboard_table import build_leaderboard_table_html
from tabarena.plot.interactive.per_dataset_explorer import build_per_dataset_explorer_html

#: Emitted by the trajectory stage next to the aggregate trajectory artifacts, and the marker
#: for "this cell gets a per-dataset browser".
_PER_DATASET_TRAJECTORIES = "tuning_trajectories_per_dataset.csv"


def process_one_folder(
    *,
    base_input_path: Path,
    base_output_path: Path,
    subset_label: str | None = None,
    dataset_metadata: pd.DataFrame | None = None,
):
    """Copy one subset's artifacts into the website layout.

    ``subset_label`` is the human-readable subset name used in the interactive
    explorers' headline (e.g. "Models only | All Tasks | Small"); omitted when ``None``.
    ``dataset_metadata`` is the benchmark's one-row-per-dataset frame, used by the per-dataset
    browser; it is the same for every subset, so the caller loads it once.
    """
    base_output_path.mkdir(parents=True, exist_ok=True)

    # N datasets file
    results_per_split = pd.read_csv(base_input_path / "results_per_split.csv", low_memory=False)
    n_datasets = len(results_per_split["dataset"].unique())
    (base_output_path / f"n_datasets_{n_datasets}").touch()

    for file_name in [
        "website_leaderboard.csv",
    ]:
        shutil.copy(
            base_input_path / file_name,
            base_output_path / file_name,
        )

    # Interactive explorers (self-contained HTML embedded by the leaderboard
    # app) and their underlying data exports. Copy-if-present so raw artifact
    # folders generated before these existed still process cleanly.
    for extra_path in [
        "pareto_front_explorer.html",
        # The train-time twin of the Pareto explorer. The website's "I care about" selector
        # switches to it for the fast-to-train view; both are rendered from the same points.
        "pareto_front_explorer_time_train.html",
        "pareto_front_points.csv",
        "winrate_explorer.html",
        "winrate_matrix.csv",
        Path("tuning_trajectories") / "placeholder_name" / "tuning_trajectories_explorer.html",
        Path("tuning_trajectories") / "placeholder_name" / "tuning_trajectories.csv",
    ]:
        src = base_input_path / extra_path
        if src.is_file():
            shutil.copy(src, base_output_path / src.name)

    # The interactive twin of `tuning-impact-elo.png`. Built here rather than in
    # the evaluation step because it needs nothing beyond the website table it
    # sits next to — so a styling fix is a re-run of the (fast) conversion step,
    # and the chart can never disagree with the table below it on the site.
    website_leaderboard = pd.read_csv(base_output_path / "website_leaderboard.csv")
    build_leaderboard_explorer_html(
        website_leaderboard,
        save_path=base_output_path / "leaderboard_overview_explorer.html",
        # No in-page heading: the website's panel header already names the
        # figure and the subset. The label identifies the standalone file.
        page_title=f"TabArena leaderboard explorer — {subset_label}"
        if subset_label
        else "TabArena leaderboard explorer",
    )
    # The leaderboard table itself, for the same reason and from the same frame:
    # the app embeds this instead of rendering its own table, so the two cannot
    # style or sort the numbers differently.
    build_leaderboard_table_html(
        website_leaderboard,
        save_path=base_output_path / "leaderboard_table.html",
        page_title=f"TabArena leaderboard table — {subset_label}" if subset_label else "TabArena leaderboard table",
    )

    # The per-dataset browser, for the cells that carry the per-dataset trajectory frame. A
    # dataset's own numbers do not depend on which other datasets share its leaderboard, so the
    # evaluation only emits that frame for the unrestricted task/dataset cell and the browser
    # filters by task and size itself (see `plot_tuning_trajectories_all`).
    trajectory_path = base_input_path / "tuning_trajectories" / "placeholder_name" / _PER_DATASET_TRAJECTORIES
    method_info_path = base_input_path / "method_info.csv"
    if trajectory_path.is_file() and method_info_path.is_file():
        build_per_dataset_explorer_html(
            results_per_split=results_per_split,
            method_info=pd.read_csv(method_info_path),
            trajectories=pd.read_csv(trajectory_path),
            dataset_metadata=dataset_metadata,
            save_path=base_output_path / "per_dataset_explorer.html",
            page_title=f"TabArena per-dataset results — {subset_label}"
            if subset_label
            else "TabArena per-dataset results",
        )
