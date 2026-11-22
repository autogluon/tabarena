"""Maintainer-only: regenerate the tabarena.ai leaderboard data for the HuggingFace Space.

This is the publishing pipeline behind the live leaderboard, **not** a user example. It builds what
the Space's ``data/`` holds, every cell of the subset grid in one pass
(:func:`tabarena.website.build.build_website`), into ``generated_website_artifacts/website_data/``:

- ``entrants_*/.../website_leaderboard.csv``, one table per cell, which the Space's API hands out
  as bulk-download URLs;
- ``n_datasets.json``, each cell's number of datasets;
- ``pages.sqlite``, every cell's leaderboard table and explorer pages in one file of about 15 MB
  (:mod:`tabarena.website.page_pack`), which the Space serves from memory.

Users who just want a leaderboard should run ``examples/plots/run_generate_main_leaderboard.py``
instead.

Publishing procedure (run this script, then do the rest manually):

Prerequisites (set up once, before pushing):
- Log in so ``git push`` is authenticated. The ``hf`` CLI ships with ``huggingface_hub``
  (``pip install -U huggingface_hub``); then run:
      hf auth login --add-to-git-credential
- Git LFS + the Git Xet extension, for ``data/pages.sqlite`` and BeyondArena's zipped PNGs
  (``data_beyondarena/``). On macOS/Linux:
      curl --proto '=https' --tlsv1.2 -sSf \
          https://raw.githubusercontent.com/huggingface/xet-core/refs/heads/main/git_xet/install.sh | sh
  (or ``brew install git-xet && git xet install``); verify with ``git xet --version``.
  The ``huggingface_hub``/``hf_xet`` package only Xet-accelerates Python uploads, not
  ``git push``. https://huggingface.co/docs/hub/en/xet/using-xet-storage#git

- Note: If you want to do a safe preview of a leaderboard release, use the private `leaderboard-testing` repo:
https://huggingface.co/spaces/TabArena/leaderboard-testing/tree/main

Steps:
1. Run this script from ``scripts/`` (a few minutes with ~100 cores).

2. Clone the leaderboard Space:
   ``git clone https://huggingface.co/spaces/TabArena/leaderboard``

3. Replace the TabArena part of its ``data/``: delete ``data/entrants_*``, ``data/n_datasets.json``
   and ``data/pages.sqlite``, then copy in the contents of
   ``generated_website_artifacts/website_data/``.

4. Commit all Space file changes and push.

5. If the push is rejected with "Repository storage limit reached (Max: 1 GB)":
   run the HEAD-aware LFS purge from within the Space checkout, then push again:
       python <tabarena>/.claude/skills/update-leaderboard/purge_stale_lfs.py --delete
   It permanently deletes only the stored LFS objects the local HEAD no longer
   references, so the revision being pushed — and the untouched data_beyondarena/
   files — keep working. Old Space revisions lose their binaries (they are
   regenerable). Do NOT use the settings-UI "select all + Remove" flow — that also
   deletes objects the live revision still needs.

``--full-figures`` additionally renders the paper figure suite of every cell (static figures,
LaTeX tables, GIF animations) into ``generated_website_artifacts/raw_website_artifacts/``. That is
paper material, not published, and takes several times as long as the website build.
"""

from __future__ import annotations

import time
from pathlib import Path

from tabarena.contexts import TabArenaContext
from tabarena.plot.tuning_trajectories.plot_pareto_over_tuning_time import plot_tuning_trajectories_all
from tabarena.website.build import build_website


def render_paper_figures(tabarena_context: TabArenaContext, save_path: Path, elo_bootstrap_rounds: int) -> None:
    """Render every cell's full figure suite and tuning trajectories into ``save_path`` (paper material)."""
    tabarena_context.evaluate_all(
        save_path=save_path,
        elo_bootstrap_rounds=elo_bootstrap_rounds,
        use_latex=False,
        use_website_folder_names=True,
        evaluator_kwargs={"figure_file_type": "png"},
        engine="ray",
        website_only=False,
    )
    plot_tuning_trajectories_all(
        tabarena_context=tabarena_context,
        fig_save_dir=save_path,
        ban_bad_methods=True,
        include_baselines=True,
        file_ext=".png",
        engine="ray",
        use_elo_method_order=False,
        focus_mode=True,
        website_only=False,
    )


if __name__ == "__main__":
    import argparse

    # See the module docstring for the full publishing procedure.
    parser = argparse.ArgumentParser(description="Regenerate the tabarena.ai leaderboard data.")
    parser.add_argument(
        "--elo-bootstrap-rounds",
        type=int,
        default=200,
        help="Elo bootstrap rounds (200 = official; 1 = fast toy run without meaningful CIs).",
    )
    parser.add_argument(
        "--full-figures",
        action="store_true",
        help="Also render every cell's paper figure suite into raw_website_artifacts/ (slow, not published).",
    )
    args = parser.parse_args()

    base_dir = Path("generated_website_artifacts")
    start = time.time()
    context = TabArenaContext()
    context.load_results(download_results="auto")
    out = build_website(context, base_dir / "website_data", elo_bootstrap_rounds=args.elo_bootstrap_rounds)
    print(f"Website data written to {out} in {time.time() - start:.0f}s")

    if args.full_figures:
        render_paper_figures(context, base_dir / "raw_website_artifacts", args.elo_bootstrap_rounds)
