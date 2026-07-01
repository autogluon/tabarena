"""Maintainer-only: regenerate and publish the BeyondArena leaderboard website artifacts.

Counterpart to ``run_generate_website_artifacts.py`` (TabArena), but for the data-foundry
BeyondArena benchmark. It (1) regenerates every per-subset figure/table from the cached BeyondArena
baselines, (2) adds the cross-subset overview figure, and (3) converts the result into the website's
folder/file layout and zips it. The artifacts are then copied into the leaderboard Space's ``data/``
directory (under the ``beyondarena`` root the BeyondArena tab reads from) and committed — see the
publishing procedure in ``run_generate_website_artifacts.py``.

How it diverges from the TabArena generator:

* **Cached baselines, not raw runs.** It uses :class:`~tabarena.contexts.BeyondArenaContext` (whose
  cached results are downloaded on demand), exactly like
  ``examples/beyondarena/run_generate_beyondarena_leaderboard.py`` — no ``BenchmarkRun`` output dirs.
* **Always the ``core`` protocol.** Every leaderboard and figure is computed on BeyondArena's
  recommended ``core`` subset (each dataset's first ``folds_to_use`` splits — already enough for
  stable rankings). Each subset dimension below is layered *on top of* ``core`` (``["core", <dim>]``);
  ``"full"`` is ``core`` with no extra filter.
* **One subset axis.** Instead of TabArena's imputation/splits/tasks/datasets grid, BeyondArena has a
  single axis of subset dimensions (task type / size bucket / feature dimensionality / feature type),
  written to ``subsets/<label>/`` — the layout the BeyondArena leaderboard tab reads.
* **Cross-subset overview figure.** Adds the per-family / per-model Elo + improvability overview
  across subsets (:func:`~tabarena.plot.subset_results.plot_subset_results`), mirroring the logic of
  ``packages/tabflow_slurm/experiments/run_eval_beyondarena.py``.

Run ``python scripts/run_generate_beyondarena_website_artifacts.py``. The generate step is
time-consuming (one ``compare`` per subset); the convert step is fast. Everything is written under
``base_dir``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tabarena.contexts import BeyondArenaContext
from tabarena.plot.subset_results import plot_subset_results
from tabarena.website.process_pngs import process_png_bulk

# Subset axis of the BeyondArena tab: label -> extra predicate(s) layered on top of the always
# present "core" protocol. Labels match ``plot_subset_results``' DEFAULT_SUBSET_ORDER so the overview
# figure lays them out in the intended Task / Scale / Dimensionality-&-Features order with "full"
# last. "full" == core across all datasets (no extra filter).
BEYOND_SUBSETS: dict[str, list[str]] = {
    "full": [],
    # split regime (the beyond-IID axis)
    "random": ["random"],
    "temporal": ["temporal"],
    "grouped": ["grouped"],
    # size buckets (on max_train_rows)
    "tiny": ["tiny"],
    "small": ["small"],
    "medium": ["medium"],
    "large": ["large"],
    # feature dimensionality / type
    "low-dim": ["low-dim"],
    "high-dim": ["high-dim"],
    "text": ["text"],
    "high-cardinality": ["high-cardinality"],
}

# Per-subset figures produced by ``compare`` that the BeyondArena tab renders. NOTE: unlike TabArena
# there is no HPO tuning-trajectory figure (``pareto_n_configs_imp``) — BeyondArena is evaluated on a
# single ``core`` protocol, not a tuning sweep, so ``plot_tuning_trajectories_all`` is not run.
_SUBSET_FIGURES = (
    "tuning-impact-elo",
    "pareto_front_improvability_vs_time_infer",
    "winrate_matrix",
)

# Methods highlighted as "contenders" in the overview figure (their own line in the per-family plot,
# star-marked in the per-model plot). Leave empty for the neutral official leaderboard.
CONTENDER_MODELS: list[str] = []


class BeyondArenaWebsiteArtifactGenerator:
    """Regenerate, convert, and zip the BeyondArena leaderboard website artifacts.

    All output (both subfolders and both zips) is written under ``base_dir``. The convert step reads
    what the generate step wrote, so the two share the raw artifacts subfolder.
    """

    def __init__(
        self,
        base_dir: str | Path,
        raw_artifacts_dirname: str = "raw_website_artifacts",
        clean_artifacts_dirname: str = "clean_website_artifacts",
    ):
        self.base_dir = Path(base_dir)
        self.raw_artifacts_dir = self.base_dir / raw_artifacts_dirname
        self.clean_artifacts_dir = self.base_dir / clean_artifacts_dirname

    def generate_website_artifacts(self):
        figure_file_type = "png"

        context = BeyondArenaContext()
        # Load the cached baselines once and reuse them for every subset (avoids re-resolving results
        # per subset). Each subset is then filtered from this frame inside ``compare``.
        ta_results = context.load_results(download_results="auto")

        leaderboards = {}
        for label, extra in BEYOND_SUBSETS.items():
            subset = ["core", *extra]  # ALWAYS core — the recommended BeyondArena evaluation protocol.
            out_dir = self.raw_artifacts_dir / "subsets" / label
            print(f"\n############### Evaluating subset: {label} (subset={subset})")

            leaderboard = context.compare(
                output_dir=out_dir,
                ta_results=ta_results,
                subset=subset,
                figure_file_type=figure_file_type,
                # Needed by the overview figure (per-subset N) and the per-subset n_datasets marker.
                add_dataset_count=True,
            )

            # The website leaderboard CSV the tab renders (Type/TypeName/Model/Elo/... columns).
            website = context.leaderboard_to_website_format(leaderboard, include_type=True)
            website.to_csv(out_dir / "website_leaderboard.csv", index=False)

            # n_datasets marker (the tab shows this count); carried by ``add_dataset_count=True``.
            n_datasets = int(leaderboard["n_datasets_total"].iloc[0])
            (out_dir / f"n_datasets_{n_datasets}").touch()

            leaderboards[label] = leaderboard

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
        for label, lb in leaderboards.items():
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

        # Place the zip next to (and named after) the raw artifacts folder.
        shutil.make_archive(str(self.raw_artifacts_dir), "zip", root_dir=self.raw_artifacts_dir)

    def convert_to_website_format(self):
        input_path = self.raw_artifacts_dir
        output_path = self.clean_artifacts_dir
        figure_file_type = "png"

        # -- Per-subset folders: copy the CSV + n_datasets marker, copy the rendered figures.
        for subset_dir in sorted((input_path / "subsets").iterdir()):
            if not subset_dir.is_dir():
                continue
            out_dir = output_path / "subsets" / subset_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)

            shutil.copy(subset_dir / "website_leaderboard.csv", out_dir / "website_leaderboard.csv")
            for marker in subset_dir.glob("n_datasets_*"):
                (out_dir / marker.name).touch()

            for fig in _SUBSET_FIGURES:
                src = subset_dir / f"{fig}.{figure_file_type}"
                if src.exists():
                    shutil.copy(src, out_dir / f"{fig}.{figure_file_type}")
                else:
                    print(f"WARNING: expected figure not found, skipping: {src}")

        # -- Cross-subset overview figures (per_family_*/per_model_* elo & improvability).
        overview_in = input_path / "result_plots"
        overview_out = output_path / "result_plots"
        overview_out.mkdir(parents=True, exist_ok=True)
        for png in sorted(overview_in.glob("*.png")):
            shutil.copy(png, overview_out / png.name)

        # -- Dataset-subset memberships, if present (a handy sidecar for the tab / debugging).
        subsets_json = input_path / "dataset_subsets.json"
        if subsets_json.exists():
            shutil.copy(subsets_json, output_path / subsets_json.name)

        # Zip every PNG (subset figures + overview) into <name>.png.zip and drop the raw PNGs, matching
        # what the leaderboard app expects (it lazily unzips on demand).
        process_png_bulk(path=output_path)

        # Place the zip next to (and named after) the clean artifacts folder.
        shutil.make_archive(str(output_path), "zip", root_dir=output_path)


if __name__ == "__main__":
    # Everything (both subfolders and both zips) is written under base_dir.
    generator = BeyondArenaWebsiteArtifactGenerator(base_dir=Path("generated_beyondarena_website_artifacts"))

    # Generate the 'raw_website_artifacts' folder (time-consuming: one compare per subset).
    generator.generate_website_artifacts()

    # Generate the 'clean_website_artifacts' folder (fast).
    generator.convert_to_website_format()
