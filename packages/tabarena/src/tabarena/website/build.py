"""Build the leaderboard Space's ``data/`` in one pass: every cell's numbers, explorers and pages.

The website shows one leaderboard per cell of the subset grid (:mod:`tabarena.evaluation.subset_grid`).
:func:`build_website` runs each cell as one task that evaluates it (:func:`evaluate_single`), draws
its tuning trajectories (:func:`plot_tuning_trajectories`) and renders its pages
(:func:`process_one_folder`) in a scratch folder that is gone when the task ends. The task hands
back the cell's dataset count and its pages, already cut into the pieces the page pack stores. The
output folder is what the Space's ``data/`` holds:

- ``<cell>/website_leaderboard.csv`` for every cell, the files the Space's API hands out for download;
- ``n_datasets.json``, each cell's number of datasets (:func:`write_n_datasets_index`);
- ``pages.sqlite``, every cell's pages in one file (:mod:`tabarena.website.page_pack`).

Three things keep it fast. Pools that admit the same methods (:func:`group_identical_pools`) are
computed once and published under each of their keys. The tuning-trajectory inputs depend on the
pool alone, so each pool's are prepared once and shared through ray's object store, where every cell
used to load all of them again. And nothing the website does not show is computed: no static
figure, no printed leaderboard, no validation-score leaderboard of the trajectories.
"""

from __future__ import annotations

import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from tqdm import tqdm

from tabarena.evaluation.entrants import ENTRANT_POOLS, group_identical_pools
from tabarena.evaluation.eval_all import evaluate_single
from tabarena.evaluation.subset_grid import get_all_subset_combinations, get_website_folder_name
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import (
    _prefetch_hpo_trajectories,
    _prepare_tuning_trajectories_data,
    plot_tuning_trajectories,
)
from tabarena.website.page_pack import PAGES_FILE, chunk_pages, write_page_pack
from tabarena.website.process_artifacts_to_website import process_one_folder, write_n_datasets_index

if TYPE_CHECKING:
    from collections.abc import Iterator

    import pandas as pd

    from tabarena.contexts import TabArenaContext

# Subset folder segment -> the human label the website shows. Mirrors ``data_loading.TASK_LABELS`` /
# ``DATASET_LABELS`` in the leaderboard Space. The entrant segments come from ``ENTRANT_POOLS`` rather
# than a copy of the keys: there is one pool per combination of the system categories, so a
# hand-written list goes stale as soon as a category is added, and ``cell_label`` drops what it
# cannot name.
CELL_LABELS: dict[str, str] = {
    **{f"entrants_{pool.key}": pool.label for pool in ENTRANT_POOLS},
    "tasks_all": "All Tasks",
    "tasks_classification": "Classification",
    "tasks_regression": "Regression",
    "tasks_binary": "Binary",
    "tasks_multiclass": "Multiclass",
    "datasets_all": "All Datasets",
    "datasets_small": "Small",
    "datasets_medium": "Medium",
    "datasets_balanced": "Balanced",
    "datasets_imbalanced": "Imbalanced",
    "datasets_extreme": "Extreme imbalance",
    "splits_lite": "Lite",
    "imputation_no": "no imputation",
}

#: The evaluation settings of every published cell (``evaluate_all`` builds the same).
EVALUATOR_KWARGS: dict = {"use_latex": False, "banned_pareto_methods": ["KNN", "LR"], "figure_file_type": "png"}


def cell_label(rel_path: Path) -> str:
    """Human-readable label for a cell folder, e.g. "Models only | All Tasks | Small".

    Only the parts that carry information are shown: the default imputation and splits settings
    are dropped, so a path like ``entrants_models/imputation_yes/splits_all/tasks_all/datasets_small``
    reads as "<entrants> | <tasks> | <datasets>". The entrant pool is always shown, since which
    numbers a figure reports depends on who competed, so an unknown ``entrants_*`` segment is an
    error rather than something to drop silently.
    """
    entrants = next((part for part in rel_path.parts if part.startswith("entrants_")), None)
    if entrants is not None and entrants not in CELL_LABELS:
        raise KeyError(f"No label for entrant pool {entrants!r}; known: {sorted(CELL_LABELS)}")
    return " | ".join(CELL_LABELS[part] for part in rel_path.parts if part in CELL_LABELS)


@dataclass(frozen=True)
class WebsiteCell:
    """One cell to compute, with the folders it is published under: one per pool of its twin group."""

    entrant_pool: str
    use_imputation: bool
    problem_type: str
    with_baselines: bool
    dataset_subset: str | None
    lite: bool
    average_seeds: bool
    folders: tuple[str, ...]

    @property
    def subset(self) -> list[str]:
        """The task filters of this cell, as ``subset_tasks`` takes them."""
        subset = [self.problem_type] if self.problem_type != "all" else []
        if self.dataset_subset is not None:
            subset.append(self.dataset_subset)
        if self.lite:
            subset.append("lite")
        return subset


def website_cells(pool_groups: list[list[str]]) -> list[WebsiteCell]:
    """Every cell of the grid, computed for the first pool of each group and published for all of them."""
    twins = {group[0]: group for group in pool_groups}
    cells = []
    for (
        pool,
        use_imputation,
        problem_type,
        with_baselines,
        dataset_subset,
        lite,
        average_seeds,
    ) in get_all_subset_combinations():
        if pool not in twins:
            continue
        folders = tuple(
            get_website_folder_name(
                entrant_pool=twin,
                use_imputation=use_imputation,
                problem_type=problem_type,
                dataset_subset=dataset_subset,
                lite=lite,
            )
            .relative_to("website_data")
            .as_posix()
            for twin in twins[pool]
        )
        cells.append(
            WebsiteCell(
                entrant_pool=pool,
                use_imputation=use_imputation,
                problem_type=problem_type,
                with_baselines=with_baselines,
                dataset_subset=dataset_subset,
                lite=lite,
                average_seeds=average_seeds,
                folders=folders,
            )
        )
    return cells


def build_cell(
    cell: WebsiteCell,
    *,
    tabarena_context: TabArenaContext,
    df_results: pd.DataFrame,
    method_metadata_info: pd.DataFrame,
    dataset_metadata: pd.DataFrame,
    trajectory_inputs: tuple[pd.DataFrame, pd.DataFrame],
    output_dir: Path,
    elo_bootstrap_rounds: int,
) -> tuple[dict[str, int], dict[str, list[bytes]], dict[bytes, bytes]]:
    """Compute one cell and publish it under each of its folders.

    Writes each folder's ``website_leaderboard.csv`` under ``output_dir`` and returns the dataset
    count per folder and the folders' pages as :func:`chunk_pages` gives them. A cell without a
    dataset or without a method returns nothing, as the evaluation skips it.
    """
    with tempfile.TemporaryDirectory(prefix="tabarena_cell_") as tmp:
        raw = Path(tmp) / "raw"
        evaluate_single(
            tabarena_context=tabarena_context,
            df_results=df_results,
            use_imputation=cell.use_imputation,
            problem_type=cell.problem_type,
            with_baselines=cell.with_baselines,
            dataset_subset=cell.dataset_subset,
            lite=cell.lite,
            average_seeds=cell.average_seeds,
            eval_save_path=Path(tmp),
            evaluator_kwargs=EVALUATOR_KWARGS,
            method_metadata_info=method_metadata_info,
            entrant_pool=cell.entrant_pool,
            elo_bootstrap_rounds=elo_bootstrap_rounds,
            custom_folder_name=raw.name,
            website_only=True,
            static_figures="uncovered",
            verbose=False,
        )
        if not (raw / "website_leaderboard.csv").is_file():
            return {}, {}, {}
        plot_tuning_trajectories(
            tabarena_context=tabarena_context,
            subset_map={"placeholder_name": cell.subset},
            fig_save_dir=raw / "tuning_trajectories",
            average_seeds=cell.average_seeds,
            exclude_imputed=not cell.use_imputation,
            # Drops the baselines (KNN/Linear) only; weak non-baseline methods stay, greyed out.
            ban_bad_methods=True,
            # Systems compete in the pools that admit them, each as the single point it is.
            include_baselines=True,
            file_ext=".png",
            use_elo_method_order=False,
            focus_mode=True,
            website_only=True,
            static_figures="uncovered",
            entrant_pool=cell.entrant_pool,
            # A dataset's own trajectory does not depend on the other datasets of the cell, so the
            # per-dataset browser reads it from the unrestricted cell and filters by itself.
            per_dataset_trajectories=cell.problem_type == "all" and cell.dataset_subset is None,
            prepared=trajectory_inputs,
        )
        counts, pages = {}, {}
        for folder in cell.folders:
            staged = Path(tmp) / "pages" / folder
            counts[folder] = process_one_folder(
                base_input_path=raw,
                base_output_path=staged,
                subset_label=cell_label(Path(folder)),
                dataset_metadata=dataset_metadata,
            )
            (output_dir / folder).mkdir(parents=True, exist_ok=True)
            shutil.copy(staged / "website_leaderboard.csv", output_dir / folder / "website_leaderboard.csv")
            pages.update({f"{folder}/{page.name}": page.read_bytes() for page in sorted(staged.glob("*.html"))})
    layout, pieces = chunk_pages(pages)
    return counts, layout, pieces


def build_website(
    tabarena_context: TabArenaContext,
    output_dir: str | Path,
    *,
    elo_bootstrap_rounds: int = 200,
    engine: str = "ray",
    progress_bar: bool = True,
) -> Path:
    """Build the Space's ``data/`` for ``tabarena_context``'s results into ``output_dir``; return the folder.

    ``output_dir`` is emptied first, so nothing an earlier build wrote can survive into the Space.
    ``engine="sequential"`` runs every cell in this process, for debugging. ``elo_bootstrap_rounds=1``
    gives a quick toy build (its Elo CIs are meaningless); 200 is the official setting.
    """
    output_dir = Path(output_dir)
    shutil.rmtree(output_dir, ignore_errors=True)
    output_dir.mkdir(parents=True)

    df_results = tabarena_context.load_results()
    df_results = tabarena_context.fillna_metrics(
        df_to_fill=df_results,
        df_fillna=df_results[df_results["method"] == tabarena_context.fillna_method],
    )
    df_results = df_results.copy(deep=True)
    if "imputed" not in df_results.columns:
        df_results["imputed"] = False
    df_results["imputed"] = df_results["imputed"].fillna(0)
    method_metadata_info = tabarena_context.method_metadata_collection.info()

    cells = website_cells(group_identical_pools(method_metadata_info))
    _prefetch_hpo_trajectories(tabarena_context.method_metadata_collection.method_metadata_lst)
    shared = {
        "tabarena_context": tabarena_context,
        "df_results": df_results,
        "method_metadata_info": method_metadata_info,
        "dataset_metadata": tabarena_context.task_metadata_collection.per_dataset_frame(),
        "output_dir": output_dir,
        "elo_bootstrap_rounds": elo_bootstrap_rounds,
    }
    if engine == "sequential":
        results = _build_sequential(cells, shared)
    elif engine == "ray":
        results = _build_with_ray(cells, shared)
    else:
        raise ValueError(f"engine must be 'ray' or 'sequential', not {engine!r}")

    counts: dict[str, int] = {}
    layout: dict[str, list[bytes]] = {}
    pieces: dict[bytes, bytes] = {}
    for cell_counts, cell_layout, cell_pieces in tqdm(
        results, total=len(cells), desc="Building the website cells", disable=not progress_bar, mininterval=1
    ):
        counts.update(cell_counts)
        layout.update(cell_layout)
        for digest, piece in cell_pieces.items():
            pieces.setdefault(digest, piece)
    write_n_datasets_index(output_dir, counts)
    write_page_pack(output_dir / PAGES_FILE, layout, pieces)
    return output_dir


def _prepare_pool(tabarena_context: TabArenaContext, pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """The tuning-trajectory inputs of one pool, as :func:`plot_tuning_trajectories` would prepare them."""
    return _prepare_tuning_trajectories_data(
        tabarena_context=tabarena_context,
        extra_results=None,
        methods_to_display=None,
        include_baselines=True,
        entrant_pool=pool,
    )


def _build_sequential(cells: list[WebsiteCell], shared: dict) -> Iterator[tuple]:
    prepared: dict[str, tuple] = {}
    for cell in cells:
        if cell.entrant_pool not in prepared:
            prepared = {cell.entrant_pool: _prepare_pool(shared["tabarena_context"], cell.entrant_pool)}
        yield build_cell(cell, trajectory_inputs=prepared[cell.entrant_pool], **shared)


#: Per ray worker: the most recently used pool's trajectory inputs, by object-store reference.
_TRAJECTORY_CACHE: dict[str, tuple] = {}


def _prepare_task(shared_ref: list, pool: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    from tabarena.utils.parallel_for import _resolve_ray_context

    return _prepare_pool(_resolve_ray_context(shared_ref[0])["tabarena_context"], pool)


def _build_task(cell: WebsiteCell, shared_ref: list, prepared_ref: list) -> tuple:
    import ray

    from tabarena.utils.parallel_for import _resolve_ray_context

    # Both references arrive inside lists, so ray hands them over unresolved: a worker deserializes
    # the shared inputs once, and a pool's trajectory inputs once per run of that pool's cells,
    # rather than both for every task.
    key = prepared_ref[0].hex()
    if key not in _TRAJECTORY_CACHE:
        _TRAJECTORY_CACHE.clear()
        _TRAJECTORY_CACHE[key] = ray.get(prepared_ref[0])
    return build_cell(cell, trajectory_inputs=_TRAJECTORY_CACHE[key], **_resolve_ray_context(shared_ref[0]))


def _build_with_ray(cells: list[WebsiteCell], shared: dict) -> Iterator[tuple]:
    import ray

    from tabarena.utils.shipping import shipping_context

    if not ray.is_initialized():
        ray.init()
    prepare, build = ray.remote(_prepare_task), ray.remote(_build_task)
    with shipping_context():
        shared_ref = ray.put(shared)
    pools = list(dict.fromkeys(cell.entrant_pool for cell in cells))
    prepared = {pool: prepare.remote([shared_ref], pool) for pool in pools}
    pending = [build.remote(cell, [shared_ref], [prepared[cell.entrant_pool]]) for cell in cells]
    # In completion order, so the driver merges each cell's pieces as they arrive instead of
    # holding every cell's result at once.
    while pending:
        done, pending = ray.wait(pending, num_returns=min(len(pending), 8), timeout=1.0)
        for ref in done:
            yield ray.get(ref)
