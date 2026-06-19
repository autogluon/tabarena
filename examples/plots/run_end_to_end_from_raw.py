"""Run the full TabArena pipeline on raw result artifacts and compare to the leaderboard.

This is the "bring your own results" workflow. Given a directory of raw per-run
``results.pkl`` artifacts, ``EndToEndSingle.from_path_raw`` (1) infers the method
metadata, (2) caches the raw artifacts, (3) generates and caches the *processed*
predictions, and (4) simulates HPO + ensembling to produce the *results* -- the same
raw -> processed -> results pipeline TabArena runs for every method. The resulting
method is then registered in ``TabArenaContext`` and compared against the official
leaderboard (figures + tables are written to ``fig_output_dir``; missing values are
imputed to the default RandomForest).

As a runnable stand-in for your own method, this downloads the *public* raw artifacts
of TabPFN-3 (~120 MB) from TabArena's R2 storage and regenerates them locally under a
distinct name (via ``name_suffix``), so the reproduced method appears *alongside* the
official TabPFN-3 entry instead of colliding with it. Point ``url`` / ``path_raw`` at
your own raw artifacts to benchmark a new method instead.

Note: regenerating processed + results from raw is compute-heavy (HPO simulation across
all TabArena tasks), so expect this to run for a while. Every artifact is cached under
``~/.cache/tabarena/``, so subsequent ``compare`` calls are fast.
"""

from __future__ import annotations

from pathlib import Path

from tabarena.nips2025_utils.artifacts.download_utils import download_and_extract_zip
from tabarena.nips2025_utils.end_to_end_single import EndToEndSingle
from tabarena.nips2025_utils.tabarena_context import TabArenaContext

if __name__ == "__main__":
    # Public raw artifacts of TabPFN-3, used here as a stand-in for "your own method".
    # Swap `url` / `path_raw` for your own raw `results.pkl` artifacts to benchmark a new method.
    url = "https://data.tabarena.ai/cache/artifacts/tabarena-2026-05-13/methods/TabPFN-3/raw.zip"
    path_raw = Path("local_data") / "TabPFN-3-raw"
    fig_output_dir = Path("tabarena_figs") / "TabPFN-3-reproduced"

    download = True
    if download:
        download_and_extract_zip(url=url, path_local=path_raw)

    # Run raw -> processed -> results end-to-end and cache every artifact under ~/.cache/tabarena/.
    # `name_suffix` renames the regenerated method (and its configs) so it does not collide with
    # the official TabPFN-3 already on the leaderboard. Run once; the cache makes re-runs cheap.
    end_to_end = EndToEndSingle.from_path_raw(path_raw=path_raw, name_suffix="_reproduced")

    # Compare the reproduced method against the full official TabArena leaderboard.
    context = TabArenaContext(extra_methods=[end_to_end.method_metadata])
    leaderboard = context.compare(output_dir=fig_output_dir)
    print(leaderboard)
