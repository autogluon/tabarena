"""Maintainer-only: regenerate and publish the tabarena.ai website artifacts.

This is the publishing pipeline behind the live leaderboard, **not** a user
example. It (1) regenerates every per-subset figure/table + tuning trajectory
(time-consuming), (2) converts them into the website's folder/file layout, and
(3) zips the result. The artifacts are then manually copied into the HuggingFace
Space ``data/`` directory and committed.

Users who just want a leaderboard should run
``examples/plots/run_generate_main_leaderboard.py`` instead.

Publishing procedure (run this script, then do the rest manually):

Prerequisites (set up once, before pushing):
- Install Git and Git LFS, then install the Git Xet extension so the large image
  files push efficiently to Xet storage. This git workflow needs the Git Xet
  extension specifically -- the ``huggingface_hub``/``hf_xet`` package only
  Xet-accelerates Python uploads, not ``git push``. On macOS/Linux:
      curl --proto '=https' --tlsv1.2 -sSf \
          https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh
  (or ``brew install git-xet && git xet install``); verify with ``git xet --version``.
  https://huggingface.co/docs/hub/en/xet/using-xet-storage#git
- Log in so ``git push`` is authenticated. The ``hf`` CLI ships with
  ``huggingface_hub`` (``pip install -U huggingface_hub``); then run:
      hf auth login --add-to-git-credential
  (the legacy ``huggingface-cli login`` still works but warns it is deprecated).

- Note: If you want to do a safe preview of a leaderboard release, use the private `leaderboard-testing` repo:
https://huggingface.co/spaces/TabArena/leaderboard-testing/tree/main

Steps:
1. Run this script to generate the 'raw_website_artifacts' folder
   (time-consuming, with 192 cores it takes a few minutes) and then the
   'clean_website_artifacts' folder (fast).

2. Clone the leaderboard Space:
   ``git clone https://huggingface.co/spaces/TabArena/leaderboard``

3. Replace its ``data/`` folder: delete the existing ``data/``, then copy in the
   contents of 'generated_website_artifacts/clean_website_artifacts/website_data/'
   (or unzip clean_website_artifacts.zip in that location).

4. Commit all Space file changes and push.

5. If the push is still rejected, check the "Git Storage Usage" here:
   https://huggingface.co/spaces/TabArena/leaderboard/settings
   If it is >700 MB, you may have to delete the stored files to free up space.
   NOTE: This is a destructive operation, the old leaderboard will cease to
   function once this is done. To delete the stored files, go here and click the
   top-left button to check all files, then click "Remove selected":
   https://huggingface.co/spaces/TabArena/leaderboard/settings?lfs-files=true
"""

from __future__ import annotations

import shutil
from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories_all
from tabarena.website.process_artifacts_to_website import process_one_folder
from tabarena.website.process_pngs import process_png_bulk


class WebsiteArtifactGenerator:
    """Regenerate, convert, and zip the tabarena.ai website artifacts.

    All output (both subfolders and both zips) is written under ``base_dir``.
    The convert step reads what the generate step wrote, so the two share the
    raw artifacts subfolder.
    """

    def __init__(
        self,
        base_dir: str | Path,
        raw_artifacts_dirname: str = "raw_website_artifacts",
        clean_artifacts_dirname: str = "clean_website_artifacts",
    ):
        self.base_dir = Path(base_dir)
        # Subfolders under base_dir for the raw figures/tables and the
        # website-formatted artifacts.
        self.raw_artifacts_dir = self.base_dir / raw_artifacts_dirname
        self.clean_artifacts_dir = self.base_dir / clean_artifacts_dirname

    def generate_website_artifacts(
        self,
        *,
        run_evaluate: bool = True,
        run_trajectories: bool = True,
        elo_bootstrap_rounds: int = 200,
        zip_raw: bool = False,
        website_only: bool = True,
    ):
        """Regenerate the raw figures/tables.

        ``run_evaluate`` / ``run_trajectories`` allow partial refreshes when only
        one pipeline's outputs changed (e.g. a trajectory-styling fix): the other
        phase's existing raw artifacts are left in place and the convert step
        picks them up unchanged. ``elo_bootstrap_rounds=1`` gives a fast toy run
        (Elo CIs are meaningless then); 200 is the official setting. ``zip_raw``
        additionally zips the raw artifacts folder (~hundreds of MB, several
        minutes) — the publishing flow only needs the *clean* zip, so this is
        off by default. ``website_only`` (default) renders only the outputs the
        website ships per subset instead of the full paper figure suite
        (~35 figures per subset, incl. GIF animations); disable it when the raw
        artifacts should double as paper material.
        """
        save_path = self.raw_artifacts_dir  # folder to save all figures and tables

        # Set to True if you have the appropriate latex packages installed for nicer figure style
        # TODO: use_latex=True makes some of the plots look worse, but makes the tuning-impact-elo figures look better.
        #  To avoid needing 2 people to compute this properly, we should selectively disable `use_latex` for certain figures.
        #  Alternatively, make all figures without `use_latex` look nice, and drop the latex logic entirely.
        use_latex: bool = False
        download_results = "auto"  # Set to False to avoid re-download

        run_ablations = False
        figure_file_type = "png"

        # Generate the per-subset-combination figures/tables and tuning trajectories in
        # parallel (one job per combination). "ray" parallelizes across all local CPUs;
        # set to "sequential" to debug or "auto" to follow the TabArenaContext backend.
        engine = "ray"

        tabarena_context = TabArenaContext()
        tabarena_context.load_results(download_results=download_results)

        evaluator_kwargs = {
            "figure_file_type": figure_file_type,
        }
        file_ext = f".{figure_file_type}"

        if run_ablations:
            tabarena_context.plot_runtime_per_method(
                save_path=Path(save_path) / "ablation" / "all-runtimes",
            )

            tabarena_context.generate_per_dataset_tables(
                save_path=Path(save_path) / "per_dataset",
            )

        if run_evaluate:
            tabarena_context.evaluate_all(
                save_path=save_path,
                elo_bootstrap_rounds=elo_bootstrap_rounds,
                use_latex=use_latex,
                use_website_folder_names=True,
                evaluator_kwargs=evaluator_kwargs,
                engine=engine,
                website_only=website_only,
            )

        if run_trajectories:
            plot_tuning_trajectories_all(
                tabarena_context=tabarena_context,
                fig_save_dir=save_path,
                # Drops the baselines (KNN/Linear) only; weak non-baseline
                # methods stay, greyed out by the focus styling.
                ban_bad_methods=True,
                file_ext=file_ext,
                engine=engine,
                # Order methods per-plot by each plot's own y-axis instead of pinning a
                # single Elo-derived order across the whole set (e.g. so the improvability
                # legend ends with the lowest/best method).
                use_elo_method_order=False,
                # Website styling: Pareto-front methods in family colors with direct
                # labels, all other trajectories greyed out. Also writes the
                # interactive tuning_trajectories_explorer.html per subset.
                focus_mode=True,
                website_only=website_only,
            )

        if zip_raw:
            # Place the zip next to (and named after) the raw artifacts folder.
            shutil.make_archive(
                str(save_path),
                "zip",
                root_dir=save_path,
            )

    def convert_to_website_format(self):
        input_path = self.raw_artifacts_dir
        output_path = self.clean_artifacts_dir

        file_paths = input_path.glob("**/website_leaderboard.csv")

        for path in file_paths:
            base_input_path = Path(path).parent
            base_output_path = output_path / base_input_path.relative_to(
                input_path,
            )
            process_one_folder(
                base_input_path=Path(path).parent,
                base_output_path=base_output_path,
            )
            process_png_bulk(path=base_output_path)

        # Place the zip next to (and named after) the clean artifacts folder.
        shutil.make_archive(
            str(output_path),
            "zip",
            root_dir=output_path / "website_data",
        )


if __name__ == "__main__":
    import argparse

    # See the module docstring for the full publishing procedure.
    parser = argparse.ArgumentParser(
        description="Regenerate the tabarena.ai website artifacts.",
    )
    parser.add_argument(
        "--skip-evaluate",
        action="store_true",
        help="Skip the per-subset evaluation figures/tables (reuse existing raw artifacts).",
    )
    parser.add_argument(
        "--skip-trajectories",
        action="store_true",
        help="Skip the tuning-trajectory figures (reuse existing raw artifacts).",
    )
    parser.add_argument(
        "--skip-convert",
        action="store_true",
        help="Skip the raw -> website-format conversion step.",
    )
    parser.add_argument(
        "--elo-bootstrap-rounds",
        type=int,
        default=200,
        help="Elo bootstrap rounds (200 = official; 1 = fast toy run without meaningful CIs).",
    )
    parser.add_argument(
        "--zip-raw",
        action="store_true",
        help="Also zip the raw artifacts folder (large + slow; the publish flow only needs the clean zip).",
    )
    parser.add_argument(
        "--full-figures",
        action="store_true",
        help="Render the full paper figure suite per subset instead of only the website-shipped outputs.",
    )
    args = parser.parse_args()

    # Everything (both subfolders and the zips) is written under base_dir.
    generator = WebsiteArtifactGenerator(base_dir=Path("generated_website_artifacts"))

    # Generate the 'raw_website_artifacts' folder (time-consuming; scales with core count).
    generator.generate_website_artifacts(
        run_evaluate=not args.skip_evaluate,
        run_trajectories=not args.skip_trajectories,
        elo_bootstrap_rounds=args.elo_bootstrap_rounds,
        zip_raw=args.zip_raw,
        website_only=not args.full_figures,
    )

    # Generate the 'clean_website_artifacts' folder (fast)
    if not args.skip_convert:
        generator.convert_to_website_format()
