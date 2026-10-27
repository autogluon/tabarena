"""Collect and preprocess data for leaderboard website.

Only the interactive artifacts ship. Every TabArena figure has a self-contained HTML explorer
that the site renders by default and that exports its own SVG / PDF / PNG on demand, so the
static PNGs were a second copy of what the reader already had: 80 of the 105 MB in the Space's
``data/``, and the only reason it needed Git LFS. They are still rendered into
``raw_website_artifacts/`` for paper use; they are simply not published.

BeyondArena's generator (``scripts/run_generate_beyondarena_website_artifacts.py``) runs every
subset through the same :func:`process_one_folder`; only its cross-subset overview figures stay
static PNGs.

Only what the leaderboard shows is published. A Hugging Face Space holds at most 20,000 files, and
every file here is repeated in every published cell (2,564 over both leaderboards, with the pools'
twins without leakage), so the data exports (``pareto_front_points.csv``, ``winrate_matrix.csv``,
``tuning_trajectories.csv``) and the train-time twin of the Pareto explorer (the explorer itself
carries the train-time axis) stay in the raw artifacts. Each cell's dataset count goes into one
index per leaderboard, ``n_datasets.json`` (:func:`write_n_datasets_index`), not a file per cell.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pandas as pd

from tabarena.plot.interactive._explorer_shared import inject_leakage
from tabarena.plot.interactive.leaderboard_explorer import _parse_model, build_leaderboard_explorer_html
from tabarena.plot.interactive.leaderboard_table import build_leaderboard_table_html
from tabarena.plot.interactive.per_dataset_explorer import (
    PER_DATASET_TRAJECTORIES_FILE,
    build_per_dataset_explorer_html,
)

#: Emitted by the trajectory stage next to the aggregate trajectory artifacts, and the marker
#: for "this cell gets a per-dataset browser".


def leakage_map(website_leaderboard: pd.DataFrame) -> dict[str, list[str]]:
    """The methods in a website table that carry a leakage warning, with their proven datasets.

    Read from the table's ``Leakage`` column (``MethodMetadata.leakage_evidence``), by the name the
    charts key a method on. Empty for a table written before the columns existed.
    """
    if "Leakage" not in website_leaderboard.columns:
        return {}
    flagged = {}
    for model, evidence, datasets in zip(
        website_leaderboard["Model"],
        website_leaderboard["Leakage"],
        website_leaderboard.get("LeakageDatasets", pd.Series("", index=website_leaderboard.index)),
        strict=True,
    ):
        if isinstance(evidence, str) and evidence:
            name, _variant, _url = _parse_model(str(model))
            flagged[name] = [d for d in str(datasets if isinstance(datasets, str) else "").split(";") if d]
    return flagged


def leader_name(website_leaderboard: pd.DataFrame) -> str | None:
    """Display name of the top row of a website leaderboard, without its link or imputed tag."""
    if website_leaderboard.empty or "Model" not in website_leaderboard.columns:
        return None
    name, variant, _ = _parse_model(str(website_leaderboard["Model"].iloc[0]))
    suffix = {"Default": " (default)", "Tuned": " (tuned)", "Tuned + Ens.": " (tuned + ensemble)"}
    return name + suffix.get(variant, "")


def process_one_folder(
    *,
    base_input_path: Path,
    base_output_path: Path,
    subset_label: str | None = None,
    dataset_metadata: pd.DataFrame | None = None,
    benchmark_name: str = "TabArena",
    size_buckets: list[tuple[str, str, float]] | None = None,
) -> int:
    """Copy one subset's artifacts into the website layout and return its number of datasets.

    The count goes into the leaderboard's index (:func:`write_n_datasets_index`).

    ``subset_label`` is the human-readable subset name used in the interactive
    explorers' headline (e.g. "Models only | All Tasks | Small"); omitted when ``None``.
    ``benchmark_name`` leads the standalone explorer files' page titles, and ``size_buckets``
    overrides the per-dataset browser's size chips (see ``build_per_dataset_explorer_html``).
    ``dataset_metadata`` is the benchmark's one-row-per-dataset frame, used by the per-dataset
    browser; it is the same for every subset, so the caller loads it once.
    """
    base_output_path.mkdir(parents=True, exist_ok=True)

    results_per_split = pd.read_csv(base_input_path / "results_per_split.csv", low_memory=False)
    n_datasets = len(results_per_split["dataset"].unique())

    for file_name in [
        "website_leaderboard.csv",
    ]:
        shutil.copy(
            base_input_path / file_name,
            base_output_path / file_name,
        )

    # Interactive explorers (self-contained HTML embedded by the leaderboard app). Copy-if-present
    # so raw artifact folders generated before these existed still process cleanly. The leaderboard
    # asks the Pareto explorer for each time axis it shows, train time included.
    for extra_path in [
        "pareto_front_explorer.html",
        "winrate_explorer.html",
        Path("tuning_trajectories") / "placeholder_name" / "tuning_trajectories_explorer.html",
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
        page_title=f"{benchmark_name} leaderboard explorer — {subset_label}"
        if subset_label
        else f"{benchmark_name} leaderboard explorer",
    )
    # The leaderboard table itself, for the same reason and from the same frame:
    # the app embeds this instead of rendering its own table, so the two cannot
    # style or sort the numbers differently.
    build_leaderboard_table_html(
        website_leaderboard,
        save_path=base_output_path / "leaderboard_table.html",
        page_title=f"{benchmark_name} leaderboard table — {subset_label}"
        if subset_label
        else f"{benchmark_name} leaderboard table",
    )

    # The per-dataset browser, for the cells that carry the per-dataset trajectory frame. A
    # dataset's own numbers do not depend on which other datasets share its leaderboard, so the
    # evaluation only emits that frame for the unrestricted task/dataset cell and the browser
    # filters by task and size itself (see `plot_tuning_trajectories_all`).
    trajectory_path = base_input_path / "tuning_trajectories" / "placeholder_name" / PER_DATASET_TRAJECTORIES_FILE
    method_info_path = base_input_path / "method_info.csv"
    if trajectory_path.is_file() and method_info_path.is_file():
        build_per_dataset_explorer_html(
            results_per_split=results_per_split,
            method_info=pd.read_csv(method_info_path),
            trajectories=pd.read_csv(trajectory_path),
            dataset_metadata=dataset_metadata,
            # The published table is already sorted by Elo, so its first row is the leader.
            default_contender=leader_name(website_leaderboard),
            size_buckets=size_buckets,
            save_path=base_output_path / "per_dataset_explorer.html",
            page_title=f"{benchmark_name} per-dataset results — {subset_label}"
            if subset_label
            else f"{benchmark_name} per-dataset results",
        )

    # Every chart of the cell marks the methods with a leakage warning, wherever it names them.
    flagged = leakage_map(website_leaderboard)
    if flagged:
        for page in base_output_path.glob("*.html"):
            page.write_text(inject_leakage(page.read_text(encoding="utf-8"), flagged), encoding="utf-8")
    return n_datasets


def write_n_datasets_index(root: Path, counts: dict[Path, int]) -> None:
    """Write ``root/n_datasets.json``: each published cell's number of datasets.

    Keyed by the cell's folder relative to ``root``, the path the leaderboard reads the cell from.
    """
    index = {Path(rel).as_posix(): int(n) for rel, n in sorted(counts.items(), key=lambda item: str(item[0]))}
    (root / "n_datasets.json").write_text(json.dumps(index, indent=0, sort_keys=True) + "\n")
