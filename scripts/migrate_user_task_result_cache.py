"""Migrate legacy hash-keyed UserTask result caches to the readable slug (maintainer tool).

UserTask results used to be keyed by the integer ``task_id`` hash; they are now keyed by
the readable :attr:`UserTask.slug`. The hash appears in two places per result, both
migrated here so previously computed (expensive) fits are reused *and* labelled correctly:

1. The cache **directory** ``data/<model>/<task_id>/...`` -> ``data/<model>/<slug>/...``.
2. The ``name`` field **inside** each ``results.pkl`` (``result["task_metadata"]["name"]``,
   e.g. ``"Task-9914417514"``), which becomes the leaderboard ``dataset`` label -> the slug.
   The numeric ``tid`` is left unchanged (it equals ``task_id`` and is used for matching).

By default this is **non-destructive**: migrated results are written to a new output
directory (``<output_dir>_migrated`` unless ``--dest-dir`` is given), leaving the original
untouched. Pass ``--in-place`` to instead rename + rewrite within the source tree.

Rewritten pkls are gzip-compressed by default (filename stays ``*.pkl``; readers detect gzip
transparently). The work is parallelized across ``(model, task)`` directories with Ray.

Only UserTask ids are migrated — plain integer OpenML task ids (e.g. TabArena v0.1) are
left untouched. Dataset / task pickles are not migrated: they re-materialize on the next run.

    python scripts/migrate_user_task_result_cache.py \
        --output-dir /path/to/workspace/output/<benchmark_name> \
        --metadata-csv /path/to/<suite>_tasks_metadata.csv \
        [--dest-dir /path/to/dest] [--in-place] [--dry-run] [--no-compress] [--num-ray-cpus auto]

``--metadata-csv`` must have a ``task_id_str`` column (e.g. a committed reference CSV).
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Iterable
from pathlib import Path
from typing import Literal

from tabarena.utils.pickle_utils import dumps_pickle, load_pickle

#: A unit of migration work: ``(src_dir, dst_dir, old_key, slug)`` as strings (Ray-picklable).
MigrationUnit = tuple[str, str, str, str]


def _migrate_task_dir(
    *,
    src_dir: Path,
    dst_dir: Path,
    old_key: str,
    slug: str,
    compress: bool,
    dry_run: bool,
    in_place: bool,
) -> int:
    """Migrate one ``(model, task)`` result directory; returns the number of pkls migrated.

    For each result pkl under ``src_dir`` whose ``task_metadata["tid"]`` matches ``old_key``
    (a safety check that it belongs to this task), rewrites ``task_metadata["name"]`` to
    ``slug`` and writes it (optionally gzip-compressed). When ``in_place`` is True the pkls
    are rewritten in ``src_dir`` and ``src_dir`` is then renamed to ``dst_dir``; otherwise
    the rewritten pkls are *copied* to ``dst_dir`` (preserving the subtask layout) and the
    source is left untouched.
    """
    n_pkls = 0
    for src_pkl in sorted(src_dir.rglob("*.pkl")):
        try:
            obj = load_pickle(src_pkl)  # transparently handles raw + gzip
        except Exception as exc:  # noqa: BLE001 — a broken/foreign pkl shouldn't abort the migration
            print(f"  WARNING: could not read {src_pkl} ({type(exc).__name__}); skipping.")
            continue

        task_metadata = obj.get("task_metadata") if isinstance(obj, dict) else None
        if not isinstance(task_metadata, dict) or str(task_metadata.get("tid")) != old_key:
            continue  # not a result for this task

        task_metadata["name"] = slug
        n_pkls += 1
        if dry_run:
            continue

        data = dumps_pickle(obj, compress=compress)
        if in_place:
            tmp_path = src_pkl.with_name(src_pkl.name + ".tmp")
            tmp_path.write_bytes(data)
            tmp_path.replace(src_pkl)  # atomic in-place update
        else:
            out_pkl = dst_dir / src_pkl.relative_to(src_dir)
            out_pkl.parent.mkdir(parents=True, exist_ok=True)
            out_pkl.write_bytes(data)

    if in_place and not dry_run and n_pkls:
        src_dir.rename(dst_dir)
    return n_pkls


def migrate_task_dirs_batch(*, units: list[MigrationUnit], compress: bool, dry_run: bool, in_place: bool) -> list[int]:
    """Migrate a batch of ``(model, task)`` directories. Batched + module-level so Ray can pickle it."""
    return [
        _migrate_task_dir(
            src_dir=Path(src_dir),
            dst_dir=Path(dst_dir),
            old_key=old_key,
            slug=slug,
            compress=compress,
            dry_run=dry_run,
            in_place=in_place,
        )
        for (src_dir, dst_dir, old_key, slug) in units
    ]


def _resolve_num_workers(num_ray_cpus: int | Literal["auto"]) -> int:
    if num_ray_cpus == "auto":
        return len(os.sched_getaffinity(0))
    return max(1, int(num_ray_cpus))


def _migrate_with_ray(
    units: list[MigrationUnit], *, compress: bool, dry_run: bool, in_place: bool, num_workers: int
) -> list[int]:
    """Run :func:`migrate_task_dirs_batch` over ``units`` in parallel with Ray."""
    import ray
    from tabarena.utils.ray_utils import ray_map_list, to_batch_list

    if not ray.is_initialized():
        ray.init(num_cpus=num_workers)

    batch_size = max(1, len(units) // (num_workers * 8) + 1)
    batched = ray_map_list(
        list_to_map=list(to_batch_list(units, batch_size)),
        func=migrate_task_dirs_batch,
        func_element_key_string="units",
        num_workers=num_workers,
        num_cpus_per_worker=1,
        func_kwargs={"compress": compress, "dry_run": dry_run, "in_place": in_place},
        track_progress=True,
        tqdm_kwargs={"desc": "Migrating result caches"},
    )
    return [count for batch in batched for count in batch]


def _default_dest_dir(output_dir: Path) -> Path:
    """Sibling ``<output_dir>_migrated`` directory used when no explicit dest is given."""
    return output_dir.parent / f"{output_dir.name}_migrated"


def migrate_user_task_result_cache(
    *,
    output_dir: str | Path,
    task_id_strs: Iterable[str],
    dest_dir: str | Path | None = None,
    in_place: bool = False,
    dry_run: bool = False,
    compress: bool = True,
    num_ray_cpus: int | Literal["auto"] = "auto",
) -> list[tuple[Path, Path]]:
    """Migrate legacy ``data/<model>/<task_id>/`` result caches to ``data/<model>/<slug>/``.

    Non-destructive by default: migrated results are written under ``dest_dir`` (defaulting
    to ``<output_dir>_migrated``) and the source is left intact. With ``in_place=True`` the
    source directories are renamed + rewritten instead. Each result pkl's ``name`` field is
    rewritten to the slug (see module docstring) and optionally gzip-compressed. Work is
    parallelized across ``(model, task)`` directories with Ray (``num_ray_cpus=1`` or a
    single unit runs sequentially without importing Ray).

    Args:
        output_dir: Source benchmark output directory (the one containing ``data/``).
        task_id_strs: ``task_id_str`` values of the tasks to migrate (e.g. the
            ``task_id_str`` column of a reference-metadata CSV). Non-UserTask ids are ignored.
        dest_dir: Destination output directory for copy mode. Defaults to
            ``<output_dir>_migrated``. Ignored when ``in_place=True``.
        in_place: Rename + rewrite within the source tree instead of copying to ``dest_dir``.
        dry_run: If True, only report the changes without performing them.
        compress: If True (default), gzip-compress the rewritten result pkls.
        num_ray_cpus: Ray workers (``"auto"`` = all available CPUs; ``1`` = sequential).

    Returns:
        The ``(src_dir, dst_dir)`` pairs that were migrated (or would be, if ``dry_run``).
    """
    from tabarena.benchmark.task.user_task import UserTask

    output_dir = Path(output_dir)
    # Legacy integer-hash key -> new slug, for UserTask ids only.
    key_to_slug: dict[str, str] = {}
    for task_id_str in task_id_strs:
        try:
            user_task = UserTask.from_task_id_str(str(task_id_str))
        except ValueError:
            continue  # not a UserTask id (e.g. a plain OpenML integer task id)
        key_to_slug[str(user_task.task_id)] = user_task.slug

    src_data = output_dir / "data"
    if not src_data.is_dir():
        return []

    dest_root = output_dir if in_place else (Path(dest_dir) if dest_dir is not None else _default_dest_dir(output_dir))
    mode = "in place" if in_place else f"copy -> {dest_root}"
    print(f"Migrating UserTask result caches under {output_dir} ({mode}).")

    # Enumerate the migratable (model, task) directories (cheap: dir-existence checks).
    units: list[MigrationUnit] = []
    for model_dir in sorted(p for p in src_data.iterdir() if p.is_dir()):
        for old_key, slug in key_to_slug.items():
            src_dir = model_dir / old_key
            dst_dir = dest_root / "data" / model_dir.name / slug
            if src_dir.is_dir() and not dst_dir.exists():
                print(f"{'[dry-run] ' if dry_run else ''}{src_dir} -> {dst_dir}")
                units.append((str(src_dir), str(dst_dir), old_key, slug))

    migrated = [(Path(src_dir), Path(dst_dir)) for (src_dir, dst_dir, _, _) in units]
    if not units:
        print("Nothing to migrate.")
        return migrated

    num_workers = _resolve_num_workers(num_ray_cpus)
    if num_workers > 1 and len(units) > 1:
        counts = _migrate_with_ray(
            units, compress=compress, dry_run=dry_run, in_place=in_place, num_workers=num_workers
        )
    else:
        counts = migrate_task_dirs_batch(units=units, compress=compress, dry_run=dry_run, in_place=in_place)

    verb = "Would migrate" if dry_run else "Migrated"
    print(f"{verb} {len(migrated)} result cache director(ies); {verb.lower()} {sum(counts)} result pkl(s).")
    return migrated


if __name__ == "__main__":
    import pandas as pd

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True, help="Source benchmark output dir (contains data/).")
    parser.add_argument("--metadata-csv", type=Path, required=True, help="CSV with a task_id_str column.")
    parser.add_argument(
        "--dest-dir",
        type=Path,
        default=None,
        help="Destination output dir for copy mode (default: <output_dir>_migrated).",
    )
    parser.add_argument(
        "--in-place",
        action="store_true",
        help="Rename + rewrite within the source tree instead of copying to a new dir.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Report changes without performing them.")
    parser.add_argument(
        "--no-compress",
        dest="compress",
        action="store_false",
        help="Keep rewritten result pkls uncompressed (default: gzip-compress them).",
    )
    parser.add_argument(
        "--num-ray-cpus",
        default="auto",
        help='Ray workers: "auto" (all CPUs, default) or an integer; 1 runs sequentially.',
    )
    args = parser.parse_args()

    num_ray_cpus = args.num_ray_cpus if args.num_ray_cpus == "auto" else int(args.num_ray_cpus)
    task_id_strs = pd.read_csv(args.metadata_csv)["task_id_str"].astype(str).unique().tolist()
    migrate_user_task_result_cache(
        output_dir=args.output_dir,
        task_id_strs=task_id_strs,
        dest_dir=args.dest_dir,
        in_place=args.in_place,
        dry_run=args.dry_run,
        compress=args.compress,
        num_ray_cpus=num_ray_cpus,
    )
