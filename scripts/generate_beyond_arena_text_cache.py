"""Generate the semantic-text embedding caches for the BeyondArena text tasks (maintainer tool).

For every BeyondArena task that carries text, this downloads/converts the task (via the
``BeyondArena`` metadata bundle), computes its semantic embeddings, and writes the per-task cache to
the canonical, encoder-versioned location
(:func:`~tabarena.benchmark.preprocessing.text_cache.text_cache_path`). This is the *producer* side;
end users instead download these caches (see
:func:`~tabarena.benchmark.task.data_foundry.text_cache.download_text_cache`).

Heavy + GPU-bound (loads the sentence-transformer encoder) and requires the optional ``data-foundry``
extra. Run once when the BeyondArena text tasks or the encoder change.

    python scripts/generate_beyond_arena_text_cache.py [--ignore-cache] [--dataset-names a b c]

In production these caches are *shipped inside each Data Foundry container* (as
``tabarena_text_cache.parquet``) and imported automatically when the dataset is materialized — see
:mod:`tabarena.benchmark.task.data_foundry.text_cache`. This script is the local (re)generation path:
it writes each cache to the canonical, encoder-versioned location
(:func:`~tabarena.benchmark.preprocessing.text_cache.text_cache_path`), useful for regenerating a
cache to upload into a container, or for purely-local use without re-downloading.
"""

from __future__ import annotations

import argparse


def generate_beyond_arena_text_caches(*, dataset_names: list[str] | None = None, ignore_cache: bool = False) -> int:
    """Generate caches for all (or the named) BeyondArena text tasks; returns the count generated."""
    from tabarena.benchmark.preprocessing.text_cache import generate_text_cache
    from tabarena.benchmark.task.metadata import BeyondArenaMetadataBundle
    from tabarena.benchmark.task.user_task import UserTask

    bundle = BeyondArenaMetadataBundle(dataset_names_to_run=dataset_names)
    task_metadata = bundle.load_task_metadata()  # downloads/converts the (filtered) tasks
    text_tasks = [ttm for ttm in task_metadata if ttm.has_text]
    print(f"Found {len(text_tasks)} BeyondArena text task(s) to cache (of {len(task_metadata)} total).")

    for ttm in text_tasks:
        user_task = UserTask.from_task_id_str(ttm.task_id_str)
        generate_text_cache(user_task, ignore_cache=ignore_cache)
    return len(text_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ignore-cache", action="store_true", help="Regenerate even if a cache already exists.")
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        default=None,
        help="Restrict to these dataset names (default: all BeyondArena text tasks).",
    )
    args = parser.parse_args()

    n = generate_beyond_arena_text_caches(dataset_names=args.dataset_names, ignore_cache=args.ignore_cache)
    print(f"Done. Generated/checked {n} text-task cache(s).")
