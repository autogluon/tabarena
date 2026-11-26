"""Maintainer-only: build the BeyondArena leaderboard's website artifacts in one pass.

Counterpart to ``run_generate_website_artifacts.py`` (TabArena), but for the data-foundry
BeyondArena benchmark, and built the same way as TabArena's (:mod:`tabarena.website.build`): each
cell of the grid is one ray task that evaluates it, draws its tuning trajectories and renders its
pages in a scratch folder that is gone when the task ends. The task hands back its leaderboard, its
dataset count and its pages, already cut into the pieces the page pack stores. The output folder is
what the leaderboard Space's ``data_beyondarena/`` holds:

- ``entrants_<pool>/<cell>/website_leaderboard.csv`` for every entrant pool and cell, the files the Space's
  API hands out;
- ``n_datasets.json``, each cell's number of datasets;
- ``pages.sqlite``, every cell's table and explorer pages in one file (:mod:`tabarena.website.page_pack`);
- ``entrants_<pool>/result_plots/*.png.zip``, each pool's cross-subset overview figures, the only static PNGs.

How it diverges from the TabArena build:

* **Cached baselines, not raw runs.** It uses :class:`~tabarena.contexts.BeyondArenaContext` (whose
  cached results are downloaded on demand), exactly like
  ``examples/beyondarena/run_generate_beyondarena_leaderboard.py``; no ``BenchmarkRun`` output dirs.
* **Always the ``core2k`` protocol.** Every leaderboard and figure is computed on BeyondArena's
  default ``core2k`` subset (each dataset's first splits from the 2,000-split cost-weighted
  allocation). Every filter below is layered *on top of* ``core2k``. The protocol is a filter, not a
  label: no figure carries its name.
* **Four filter axes, crossed.** The BeyondArena tab filters by split regime, dataset size, feature
  dimensionality / type and problem type (:data:`BEYOND_AXES`). Every leaderboard is one cell of
  their cross product, written to ``split_<s>/size_<z>/features_<f>/tasks_<t>/`` under its pool (the
  layout the tab reads). A cell no dataset falls into is skipped, and the tab says so.
* **Cross-subset overview figure.** Adds the per-family / per-model Elo + improvability overview
  across the single-filter cells (:func:`~tabarena.plot.subset_results.plot_subset_results`).
* **Entrant pools, as on TabArena.** Who competes changes everyone's numbers, so the grid and its overview are
  computed once per entrant pool (:mod:`tabarena.evaluation.entrants`): models only, each combination of the
  system categories, and each one's twin without the methods that carry a leakage warning. Pools that admit the
  same methods (:func:`~tabarena.evaluation.entrants.group_identical_pools`) are computed once and published under
  every key of the group, so a category no BeyondArena system carries costs nothing.

Run ``python scripts/run_generate_beyondarena_website_artifacts.py``; the output is
``generated_beyondarena_website_artifacts/clean_website_artifacts/``, emptied first. The tuning-trajectory
inputs depend on the pool alone, not on the cell, so they are prepared once per pool and shared through ray's
object store, and nothing the website does not show is computed (no static figure, no temp files that outlive a
cell).
"""

from __future__ import annotations

import itertools
import shutil
import tempfile
import traceback
from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.evaluation.entrants import filter_results_to_pool, get_entrant_pool, group_identical_pools
from tabarena.plot.interactive.per_dataset_explorer import BEYONDARENA_SIZE_BUCKETS
from tabarena.plot.subset_results import plot_subset_results
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import (
    _prefetch_hpo_trajectories,
    _prepare_tuning_trajectories_data,
    plot_tuning_trajectories,
)
from tabarena.utils.parallel_for import parallel_for
from tabarena.website.page_pack import PAGES_FILE, chunk_pages, write_page_pack
from tabarena.website.process_artifacts_to_website import process_one_folder, write_n_datasets_index
from tabarena.website.process_pngs import process_png_bulk
from tabarena.website.website_format import system_display_names

# The filter axes of the BeyondArena tab: axis -> {key: extra predicate(s) layered on top of the
# always present "core2k" protocol}. Each axis's first key, "all", adds no filter. The keys are the
# subset predicates' names, so the per-dataset browser's split / size / task filters (which key on
# the same metadata values) can be preselected with them directly. Order = folder order.
BEYOND_AXES: dict[str, dict[str, list[str]]] = {
    "split": {"all": [], "random": ["random"], "temporal": ["temporal"], "grouped": ["grouped"]},
    # size buckets (on max_train_rows)
    "size": {"all": [], "tiny": ["tiny"], "small": ["small"], "medium": ["medium"], "large": ["large"]},
    "features": {
        "all": [],
        "low-dim": ["low-dim"],
        "high-dim": ["high-dim"],
        "text": ["text"],
        "high-cardinality": ["high-cardinality"],
    },
    # problem type, named and ordered as TabArena's task axis
    "tasks": {
        "all": [],
        "classification": ["classification"],
        "regression": ["regression"],
        "binary": ["binary"],
        "multiclass": ["multiclass"],
    },
}

#: The axes the cross-subset overview figure walks, one filter at a time. Its labels match
#: ``plot_subset_results``' DEFAULT_SUBSET_ORDER, with the unfiltered cell as "full".
OVERVIEW_AXES = ("split", "size", "features")

# Methods highlighted as "contenders" in the overview figure (their own line in the per-family plot,
# star-marked in the per-model plot). Leave empty for the neutral official leaderboard.
CONTENDER_MODELS: list[str] = []

Cell = tuple[str, ...]


def pool_root(pool: str) -> Path:
    """The folder of one entrant pool's grid and overview, ``entrants_<pool>``."""
    return Path(f"entrants_{pool}")


def cell_rel_path(cell: Cell, axes: dict[str, dict[str, list[str]]] = BEYOND_AXES) -> Path:
    """The folder of one cell, ``split_<s>/size_<z>/features_<f>/tasks_<t>``."""
    return Path(*(f"{axis}_{key}" for axis, key in zip(axes, cell, strict=True)))


def cell_subset(cell: Cell, base_subset: list[str], axes: dict[str, dict[str, list[str]]] = BEYOND_AXES) -> list[str]:
    """The subset expression of one cell: ``base_subset`` ANDed with each axis's predicates."""
    return [*base_subset, *(p for axis, key in zip(axes, cell, strict=True) for p in axes[axis][key])]


def cell_label(cell: Cell) -> str:
    """Human-readable name of one cell for the explorers' page titles, e.g. ``random · tiny``."""
    keys = [key for key in cell if key != "all"]
    return " · ".join(keys) if keys else "all"


def overview_label(cell: Cell, axes: dict[str, dict[str, list[str]]] = BEYOND_AXES) -> str | None:
    """The overview figure's label for a cell that filters on at most one overview axis, else ``None``."""
    filtered = {axis: key for axis, key in zip(axes, cell, strict=True) if key != "all"}
    if not filtered:
        return "full"
    if len(filtered) == 1:
        ((axis, key),) = filtered.items()
        if axis in OVERVIEW_AXES:
            return key
    return None


def build_cell(
    *,
    cell: Cell,
    subset: list[str],
    pools: tuple[str, ...],
    output_dir: Path,
    context: BeyondArenaContext,
    ta_results,
    dataset_metadata,
    trajectory_inputs,
    figure_file_type: str,
):
    """Compute one cell for the first of ``pools`` and publish it under ``output_dir/entrants_<pool>/<cell>`` for each.

    ``pools`` is a group of entrant pools that admit the same methods (``ta_results`` is already filtered to them),
    so one evaluation serves them all. Evaluates the cell (leaderboard, website table, tuning trajectories) and
    renders its pages in a scratch folder, writes the cell's ``website_leaderboard.csv`` under each pool's folder
    and returns a dict with the leaderboard, the cell's dataset count and its pages as :func:`chunk_pages` gives
    them. Returns ``None`` when the cell failed. A failure is printed and its folders removed, so one cell cannot
    take the grid down with it and nothing half-built is published (the tab reports it as unpublished).
    """
    rels = [pool_root(pool) / cell_rel_path(cell) for pool in pools]
    try:
        with tempfile.TemporaryDirectory(prefix="beyondarena_cell_") as tmp:
            raw, staged = Path(tmp) / "raw", Path(tmp) / "staged"
            leaderboard = context.compare(
                output_dir=raw,
                ta_results=ta_results,
                subset=subset,
                figure_file_type=figure_file_type,
                # Needed by the overview figure (per-subset N).
                add_dataset_count=True,
                # Only the outputs the website ships, not the full paper figure suite.
                website_only=True,
                static_figures="uncovered",
                # No figure carries the subset's name: ``compare`` would title the explorers with the
                # joined subset expression ("core2k_random_tiny"), a filter and not a label.
                subset_label=None,
            )

            # The website leaderboard CSV the tab renders (Type/TypeName/Model/Elo/... columns).
            context.leaderboard_to_website_format(leaderboard, include_type=True).to_csv(
                raw / "website_leaderboard.csv", index=False
            )

            # The tuning trajectories (and, for the unfiltered cell, the per-dataset frame the browser
            # reads), written where the converter looks for them. Imputed results stay in, as on the
            # leaderboard: the foundation models that skip the >100k-row tables would vanish otherwise.
            plot_tuning_trajectories(
                tabarena_context=context,
                subset_map={"placeholder_name": subset},
                fig_save_dir=raw / "tuning_trajectories",
                exclude_imputed=False,
                ban_bad_methods=True,
                include_baselines=True,
                focus_mode=True,
                website_only=True,
                static_figures="uncovered",
                # A dataset's own trajectory does not depend on which other datasets share the cell,
                # so only the unfiltered one emits the per-dataset frame.
                per_dataset_trajectories=all(key == "all" for key in cell),
                file_ext=f".{figure_file_type}",
                entrant_pool=pools[0],
                prepared=trajectory_inputs,
            )
            count = process_one_folder(
                base_input_path=raw,
                base_output_path=staged,
                subset_label=cell_label(cell),
                dataset_metadata=dataset_metadata,
                benchmark_name="BeyondArena",
                size_buckets=BEYONDARENA_SIZE_BUCKETS,
            )
            # The pages do not depend on the folder they are published under, so every pool of the group
            # gets the same ones.
            pages = {}
            for rel in rels:
                (output_dir / rel).mkdir(parents=True, exist_ok=True)
                shutil.copy(staged / "website_leaderboard.csv", output_dir / rel / "website_leaderboard.csv")
                pages.update(
                    {f"{rel.as_posix()}/{page.name}": page.read_bytes() for page in sorted(staged.glob("*.html"))}
                )
    except Exception:
        print(f"FAILED cell {cell_label(cell)} (subset={subset}, pools={pools})")
        traceback.print_exc()
        for rel in rels:
            shutil.rmtree(output_dir / rel, ignore_errors=True)
        return None
    layout, pieces = chunk_pages(pages)
    return {"leaderboard": leaderboard, "count": count, "layout": layout, "pieces": pieces}


class BeyondArenaWebsiteArtifactGenerator:
    """Build the BeyondArena leaderboard website artifacts, ready to copy into the Space's ``data_beyondarena/``.

    Everything is written under ``base_dir / clean_artifacts_dirname``, emptied first.
    """

    def __init__(
        self,
        base_dir: str | Path,
        clean_artifacts_dirname: str = "clean_website_artifacts",
        axes: dict[str, dict[str, list[str]]] | None = None,
        base_subset: tuple[str, ...] = ("core2k",),
        engine: str = "ray",
    ):
        """Args:
        base_dir: Directory the output folder is written under.
        clean_artifacts_dirname: Name of the output folder.
        axes: The filter axes to cross (axis -> {key: extra predicate(s) layered on top of
            ``base_subset``}); defaults to :data:`BEYOND_AXES`. Must keep its axis names and their
            order, since they are the folder segments the tab reads; restrict the keys to
            generate part of the grid.
        base_subset: Subset expressions ANDed into *every* cell. Defaults to ``("core2k",)``, the
            default BeyondArena evaluation protocol; variants may append further predicates
            (e.g. ``("core2k", "!large")`` for the <=100k-train-rows generator).
        engine: How the cells are spread over the machine: ``"ray"`` (default, every local CPU)
            or ``"sequential"`` to debug; see :func:`~tabarena.utils.parallel_for.parallel_for`.
        """
        self.base_dir = Path(base_dir)
        self.clean_artifacts_dir = self.base_dir / clean_artifacts_dirname
        self.axes = BEYOND_AXES if axes is None else axes
        self.base_subset = list(base_subset)
        self.engine = engine

    def cells(self) -> list[Cell]:
        """Every cell of the grid, the unfiltered one first."""
        return list(itertools.product(*(list(keys) for keys in self.axes.values())))

    def build(self) -> Path:
        """Build the whole folder (every entrant pool's grid and overview figures, the index and the page pack)."""
        output_dir = self.clean_artifacts_dir
        shutil.rmtree(output_dir, ignore_errors=True)
        output_dir.mkdir(parents=True)

        context = BeyondArenaContext()
        # Load the cached baselines once and share them with every cell. Each cell is then
        # filtered from this frame inside ``compare``.
        all_results = context.load_results(download_results="auto")
        # One row per dataset, the same for every cell: the per-dataset browser's filters and its
        # metadata line read it.
        dataset_metadata = context.task_metadata_collection.per_dataset_frame()
        _prefetch_hpo_trajectories(context.method_metadata_collection.method_metadata_lst)
        method_metadata_info = context.method_metadata_collection.info()

        counts: dict[Path, int] = {}
        layout: dict[str, list[bytes]] = {}
        pieces: dict[bytes, bytes] = {}
        # Each group of pools that admit the same methods is computed once, for its first pool.
        for pools in group_identical_pools(method_metadata_info):
            ta_results = filter_results_to_pool(all_results, get_entrant_pool(pools[0]), method_metadata_info)
            # The tuning-trajectory inputs depend on the pool alone, not on the cell: prepare them once per pool.
            trajectory_inputs = _prepare_tuning_trajectories_data(
                tabarena_context=context,
                extra_results=None,
                methods_to_display=None,
                include_baselines=True,
                entrant_pool=pools[0],
            )
            self._build_pool_group(
                context,
                ta_results,
                tuple(pools),
                output_dir=output_dir,
                dataset_metadata=dataset_metadata,
                trajectory_inputs=trajectory_inputs,
                counts=counts,
                layout=layout,
                pieces=pieces,
            )
        write_n_datasets_index(output_dir, counts)
        write_page_pack(output_dir / PAGES_FILE, layout, pieces)

        # Zip the overview PNGs into <name>.png.zip and drop the raw PNGs, matching what the
        # leaderboard app expects (it lazily unzips on demand).
        process_png_bulk(path=output_dir)
        return output_dir

    def _build_pool_group(
        self,
        context: BeyondArenaContext,
        ta_results,
        pools: tuple[str, ...],
        *,
        output_dir: Path,
        dataset_metadata,
        trajectory_inputs,
        counts: dict[Path, int],
        layout: dict[str, list[bytes]],
        pieces: dict[bytes, bytes],
    ) -> None:
        """The grid and its overview for one group of pools that admit the same methods, from ``ta_results``."""
        figure_file_type = "png"
        cells = self.cells()
        inputs = []
        empty: list[Cell] = []
        for cell in cells:
            subset = cell_subset(cell, self.base_subset, self.axes)  # base_subset ALWAYS contains core2k.
            if context.subset_results(df_results=ta_results, subset=subset).empty:
                empty.append(cell)
                continue
            inputs.append({"cell": cell, "subset": subset})

        results = parallel_for(
            f=build_cell,
            inputs=inputs,
            context={
                "pools": pools,
                "output_dir": output_dir,
                "context": context,
                "ta_results": ta_results,
                "dataset_metadata": dataset_metadata,
                "trajectory_inputs": trajectory_inputs,
                "figure_file_type": figure_file_type,
            },
            engine=self.engine,
            desc=f"Building BeyondArena cells for {pools[0]}",
        )
        failed = [x["cell"] for x, res in zip(inputs, results, strict=True) if res is None]
        print(
            f"\nPublished {len(inputs) - len(failed)} of {len(cells)} cells for the pools {', '.join(pools)}; "
            f"{len(empty)} match no dataset."
        )
        if failed:
            print(f"FAILED ({len(failed)}): " + ", ".join(cell_label(c) for c in failed))

        overview_leaderboards = {}
        for x, res in zip(inputs, results, strict=True):
            if res is None:
                continue
            for pool in pools:
                counts[pool_root(pool) / cell_rel_path(x["cell"], self.axes)] = res["count"]
            layout.update(res["layout"])
            for digest, piece in res["pieces"].items():
                pieces.setdefault(digest, piece)
            label = overview_label(x["cell"], self.axes)
            if label is not None:
                overview_leaderboards[label] = res["leaderboard"]

        # Give the overview figure display names so its per-family lines resolve. compare() leaves
        # the method column as raw config-type names (e.g. "TA-REALMLP (tuned + ensemble)"), which do
        # not match plot_subset_results' family groups ("RealMLP", "TabM", ...). Mirror the rename that
        # evaluate_beyond_subsets applies: context config_type -> display + the BeyondArena fixups,
        # extended to the tuned / tuned+ensemble / default display suffixes.
        from tabarena.evaluation.beyond_arena_eval import DEFAULT_METHOD_RENAME_MAP

        rename = {**context.get_method_rename_map(), **DEFAULT_METHOD_RENAME_MAP}
        full_rename = {
            **rename,
            **{
                f"{k} {suffix}": f"{v} {suffix}"
                for k, v in rename.items()
                for suffix in ["(tuned + ensemble)", "(tuned)", "(default)"]
            },
        }
        renamed_leaderboards = {}
        for label, lb in overview_leaderboards.items():
            lb = lb.copy()
            lb["method"] = lb["method"].map(full_rename).fillna(lb["method"])
            renamed_leaderboards[label] = lb

        # Cross-subset overview figure (per-family / per-model Elo + improvability across subsets), drawn once and
        # published under every pool of the group.
        first = output_dir / pool_root(pools[0]) / "result_plots"
        plot_subset_results(
            renamed_leaderboards,
            first,
            metrics=("elo", "improvability"),
            contenders=CONTENDER_MODELS,
            systems=system_display_names(context.method_metadata_collection.info()),
        )
        # The overview is published as PNG only (zipped by ``process_png_bulk``).
        for pdf in first.glob("*.pdf"):
            pdf.unlink()
        for pool in pools[1:]:
            shutil.copytree(first, output_dir / pool_root(pool) / "result_plots")


if __name__ == "__main__":
    # Everything is written under base_dir.
    BeyondArenaWebsiteArtifactGenerator(base_dir=Path("generated_beyondarena_website_artifacts")).build()
