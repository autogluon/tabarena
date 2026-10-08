"""Maintainer-only: regenerate and publish the BeyondArena leaderboard website artifacts.

Counterpart to ``run_generate_website_artifacts.py`` (TabArena), but for the data-foundry
BeyondArena benchmark. It (1) regenerates every per-cell figure/table from the cached BeyondArena
baselines, (2) adds the cross-subset overview figure, and (3) converts the result into the website's
folder/file layout and zips it. Each cell is converted by TabArena's own
:func:`~tabarena.website.process_artifacts_to_website.process_one_folder`, so it ships the same
interactive explorers, embedded table and per-dataset browser as a TabArena cell; only the
cross-subset overview figures are static (zipped) PNGs.
The artifacts are then copied into the leaderboard Space's ``data_beyondarena/`` directory and
committed; see the publishing procedure in ``run_generate_website_artifacts.py``.

How it diverges from the TabArena generator:

* **Cached baselines, not raw runs.** It uses :class:`~tabarena.contexts.BeyondArenaContext` (whose
  cached results are downloaded on demand), exactly like
  ``examples/beyondarena/run_generate_beyondarena_leaderboard.py``; no ``BenchmarkRun`` output dirs.
* **Always the ``core2k`` protocol.** Every leaderboard and figure is computed on BeyondArena's
  default ``core2k`` subset (each dataset's first splits from the 2,000-split cost-weighted
  allocation). Every filter below is layered *on top of* ``core2k``.
* **Four filter axes, crossed.** The BeyondArena tab filters by split regime, dataset size, feature
  dimensionality / type and problem type (:data:`BEYOND_AXES`). Every leaderboard is one cell of
  their cross product, written to ``subsets/split_<s>/size_<z>/features_<f>/tasks_<t>/`` (the
  layout the tab reads). A cell no dataset falls into is skipped, and the tab says so.
* **Cross-subset overview figure.** Adds the per-family / per-model Elo + improvability overview
  across the single-filter cells (:func:`~tabarena.plot.subset_results.plot_subset_results`).

Run ``python scripts/run_generate_beyondarena_website_artifacts.py``. The generate step is one
``compare`` plus its tuning trajectories per cell, spread over every local CPU with ray; the
convert step is fast. Everything is written under ``base_dir``.
"""

from __future__ import annotations

import itertools
import shutil
import traceback
from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.plot.interactive.per_dataset_explorer import BEYONDARENA_SIZE_BUCKETS
from tabarena.plot.subset_results import plot_subset_results
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories
from tabarena.utils.parallel_for import parallel_for
from tabarena.website.process_artifacts_to_website import process_one_folder
from tabarena.website.process_pngs import process_png_bulk

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


def generate_cell(
    *,
    cell: Cell,
    subset: list[str],
    out_dir: Path,
    context: BeyondArenaContext,
    ta_results,
    figure_file_type: str,
):
    """Evaluate one cell into ``out_dir``: its leaderboard, website table and tuning trajectories.

    Returns the leaderboard, or ``None`` when the cell failed. A failure is printed and its folder
    removed, so one cell cannot take the grid down with it and the convert step does not publish
    half of it (the tab reports it as unpublished).
    """
    try:
        leaderboard = context.compare(
            output_dir=out_dir,
            ta_results=ta_results,
            subset=subset,
            figure_file_type=figure_file_type,
            # Needed by the overview figure (per-subset N).
            add_dataset_count=True,
            # Only the outputs the website ships, not the full paper figure suite.
            website_only=True,
        )

        # The website leaderboard CSV the tab renders (Type/TypeName/Model/Elo/... columns).
        website = context.leaderboard_to_website_format(leaderboard, include_type=True)
        website.to_csv(out_dir / "website_leaderboard.csv", index=False)

        # The tuning trajectories (and, for the unfiltered cell, the per-dataset frame the browser
        # reads), written where the converter looks for them. Imputed results stay in, as on the
        # leaderboard: the foundation models that skip the >100k-row tables would vanish otherwise.
        plot_tuning_trajectories(
            tabarena_context=context,
            subset_map={"placeholder_name": subset},
            fig_save_dir=out_dir / "tuning_trajectories",
            exclude_imputed=False,
            ban_bad_methods=True,
            include_baselines=True,
            focus_mode=True,
            website_only=True,
            # A dataset's own trajectory does not depend on which other datasets share the cell,
            # so only the unfiltered one emits the per-dataset frame.
            per_dataset_trajectories=all(key == "all" for key in cell),
            file_ext=f".{figure_file_type}",
        )
    except Exception:
        print(f"FAILED cell {cell_label(cell)} (subset={subset})")
        traceback.print_exc()
        shutil.rmtree(out_dir, ignore_errors=True)
        return None
    return leaderboard


class BeyondArenaWebsiteArtifactGenerator:
    """Regenerate, convert, and zip the BeyondArena leaderboard website artifacts.

    All output (both subfolders and the zip) is written under ``base_dir``. The convert step reads
    what the generate step wrote, so the two share the raw artifacts subfolder.
    """

    def __init__(
        self,
        base_dir: str | Path,
        raw_artifacts_dirname: str = "raw_website_artifacts",
        clean_artifacts_dirname: str = "clean_website_artifacts",
        axes: dict[str, dict[str, list[str]]] | None = None,
        base_subset: tuple[str, ...] = ("core2k",),
        engine: str = "ray",
    ):
        """Args:
        base_dir: Directory all output (both subfolders and the zip) is written under.
        raw_artifacts_dirname: Name of the generate step's output subfolder.
        clean_artifacts_dirname: Name of the convert step's output subfolder.
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
        self.raw_artifacts_dir = self.base_dir / raw_artifacts_dirname
        self.clean_artifacts_dir = self.base_dir / clean_artifacts_dirname
        self.axes = BEYOND_AXES if axes is None else axes
        self.base_subset = list(base_subset)
        self.engine = engine

    def cells(self) -> list[Cell]:
        """Every cell of the grid, the unfiltered one first."""
        return list(itertools.product(*(list(keys) for keys in self.axes.values())))

    def generate_website_artifacts(self):
        figure_file_type = "png"

        context = BeyondArenaContext()
        # Load the cached baselines once and share them with every cell. Each cell is then
        # filtered from this frame inside ``compare``.
        ta_results = context.load_results(download_results="auto")

        # The grid is regenerated as a whole, so no cell of a previous run can survive into the
        # convert step (which publishes every folder it finds).
        shutil.rmtree(self.raw_artifacts_dir / "subsets", ignore_errors=True)

        cells = self.cells()
        inputs = []
        empty: list[Cell] = []
        for cell in cells:
            subset = cell_subset(cell, self.base_subset, self.axes)  # base_subset ALWAYS contains core2k.
            if context.subset_results(df_results=ta_results, subset=subset).empty:
                empty.append(cell)
                continue
            out_dir = self.raw_artifacts_dir / "subsets" / cell_rel_path(cell, self.axes)
            inputs.append({"cell": cell, "subset": subset, "out_dir": out_dir})

        leaderboards = parallel_for(
            f=generate_cell,
            inputs=inputs,
            context={"context": context, "ta_results": ta_results, "figure_file_type": figure_file_type},
            engine=self.engine,
            desc="Evaluating BeyondArena cells",
        )
        failed = [x["cell"] for x, lb in zip(inputs, leaderboards, strict=True) if lb is None]
        print(f"\nPublished {len(inputs) - len(failed)} of {len(cells)} cells; {len(empty)} match no dataset.")
        if failed:
            print(f"FAILED ({len(failed)}): " + ", ".join(cell_label(c) for c in failed))

        overview_leaderboards = {}
        for x, lb in zip(inputs, leaderboards, strict=True):
            label = overview_label(x["cell"], self.axes)
            if label is not None and lb is not None:
                overview_leaderboards[label] = lb

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

        # Cross-subset overview figure (per-family / per-model Elo + improvability across subsets).
        plot_subset_results(
            renamed_leaderboards,
            self.raw_artifacts_dir / "result_plots",
            metrics=("elo", "improvability"),
            contenders=CONTENDER_MODELS,
        )

    def convert_to_website_format(self):
        input_path = self.raw_artifacts_dir
        output_path = self.clean_artifacts_dir

        # The website mirrors this folder, so a cell the generate step dropped must not survive here.
        shutil.rmtree(output_path / "subsets", ignore_errors=True)

        # One row per dataset, the same for every cell: the per-dataset browser's filters and its
        # metadata line read it.
        dataset_metadata = BeyondArenaContext().task_metadata_collection.per_dataset_frame()

        # -- Per-cell folders, converted exactly like a TabArena cell.
        for cell in self.cells():
            rel_path = Path("subsets") / cell_rel_path(cell, self.axes)
            if not (input_path / rel_path / "website_leaderboard.csv").is_file():
                continue
            process_one_folder(
                base_input_path=input_path / rel_path,
                base_output_path=output_path / rel_path,
                subset_label=cell_label(cell),
                dataset_metadata=dataset_metadata,
                benchmark_name="BeyondArena",
                size_buckets=BEYONDARENA_SIZE_BUCKETS,
            )

        # -- Cross-subset overview figures (per_family_*/per_model_* elo & improvability).
        overview_in = input_path / "result_plots"
        overview_out = output_path / "result_plots"
        shutil.rmtree(overview_out, ignore_errors=True)
        overview_out.mkdir(parents=True, exist_ok=True)
        for png in sorted(overview_in.glob("*.png")):
            shutil.copy(png, overview_out / png.name)

        # Zip the overview PNGs into <name>.png.zip and drop the raw PNGs, matching what the
        # leaderboard app expects (it lazily unzips on demand).
        process_png_bulk(path=output_path)

        # Place the zip next to (and named after) the clean artifacts folder.
        shutil.make_archive(str(output_path), "zip", root_dir=output_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Regenerate the BeyondArena website artifacts.")
    parser.add_argument(
        "--skip-generate",
        action="store_true",
        help="Skip the per-cell evaluation (reuse the existing raw artifacts) and only convert.",
    )
    args = parser.parse_args()

    # Everything (both subfolders and the zip) is written under base_dir.
    generator = BeyondArenaWebsiteArtifactGenerator(base_dir=Path("generated_beyondarena_website_artifacts"))

    if not args.skip_generate:
        # Generate the 'raw_website_artifacts' folder (time-consuming: one compare per cell).
        generator.generate_website_artifacts()

    # Generate the 'clean_website_artifacts' folder (fast).
    generator.convert_to_website_format()
