"""Migrate a reference-metadata CSV's task-name column to UserTask slugs (maintainer tool).

UserTask results used to be keyed by the integer ``task_id`` hash; they are now keyed by the
readable :attr:`UserTask.slug` (see ``scripts/migrate_user_task_result_cache.py``, which migrates
the result *caches*). A committed reference-metadata CSV, however, still carries the legacy
``Task-<task_id>`` value in its ``tabarena_task_name`` column. Eval matches the (slug-named)
result directories against this column, so the CSV must be migrated too — otherwise every task is
dropped as "not in task_metadata".

This recomputes ``tabarena_task_name`` from each row's ``task_id_str`` using the *same*
``UserTask.slug``, leaving non-UserTask rows (plain OpenML integer ids) unchanged. It is
non-destructive: the source CSV is read and a new CSV is written. All other columns/cells are
preserved verbatim (rewritten via the ``csv`` module, so numeric formatting is untouched).

    python scripts/migrate_user_task_metadata_csv.py \
        --src-csv /path/to/<suite>_tasks_metadata.csv \
        --dst-csv /path/to/<suite>_tasks_metadata_migrated.csv

``--src-csv`` must have ``task_id_str`` and ``tabarena_task_name`` columns. Pass ``--in-place`` to
overwrite the source (writes via a temp file, then atomically replaces it).
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def migrate_user_task_metadata_csv(
    *,
    src_csv: str | Path,
    dst_csv: str | Path,
    task_id_str_col: str = "task_id_str",
    task_name_col: str = "tabarena_task_name",
) -> int:
    """Rewrite ``task_name_col`` of a reference-metadata CSV to UserTask slugs.

    Recomputes the readable task name from each row's ``task_id_str`` via ``UserTask.slug``,
    leaving non-UserTask rows (plain OpenML integer ids) unchanged. Non-destructive: reads
    ``src_csv`` and writes ``dst_csv`` (all other columns preserved verbatim).

    Args:
        src_csv: Source reference-metadata CSV (must contain ``task_id_str_col`` and ``task_name_col``).
        dst_csv: Destination CSV path (parent dirs created as needed). May equal ``src_csv``;
            written via a temp file then atomically replaced so a crash can't truncate the source.
        task_id_str_col: Column holding the ``UserTask`` ``task_id_str`` (default ``"task_id_str"``).
        task_name_col: Readable-name column to rewrite (default ``"tabarena_task_name"``).

    Returns:
        The number of rows whose name was updated.
    """
    from tabarena.benchmark.task.user_task import UserTask

    src_csv, dst_csv = Path(src_csv), Path(dst_csv)

    slug_by_task_id_str: dict[str, str | None] = {}

    def _slug(task_id_str: str) -> str | None:
        if task_id_str not in slug_by_task_id_str:
            try:
                slug_by_task_id_str[task_id_str] = UserTask.from_task_id_str(task_id_str).slug
            except ValueError:
                slug_by_task_id_str[task_id_str] = None  # not a UserTask id; leave row unchanged
        return slug_by_task_id_str[task_id_str]

    with src_csv.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = reader.fieldnames
        if fieldnames is None or task_id_str_col not in fieldnames or task_name_col not in fieldnames:
            raise ValueError(
                f"CSV {src_csv} must contain columns '{task_id_str_col}' and '{task_name_col}' "
                f"(found: {fieldnames})."
            )
        rows = list(reader)

    n_updated = 0
    for row in rows:
        slug = _slug(row[task_id_str_col])
        if slug is not None and row[task_name_col] != slug:
            row[task_name_col] = slug
            n_updated += 1

    dst_csv.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then atomically replace, so the source can't be truncated on a crash
    # (matters when dst_csv == src_csv for the --in-place case).
    tmp_csv = dst_csv.with_name(dst_csv.name + ".tmp")
    with tmp_csv.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    tmp_csv.replace(dst_csv)

    print(f"Wrote {dst_csv} ({len(rows)} rows, {n_updated} task name(s) migrated to slug).")
    return n_updated


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src-csv", type=Path, required=True, help="Source reference-metadata CSV.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dst-csv", type=Path, default=None, help="Destination CSV path (new copy).")
    group.add_argument("--in-place", action="store_true", help="Overwrite the source CSV instead.")
    parser.add_argument(
        "--task-id-str-col", default="task_id_str", help="Column holding the UserTask task_id_str."
    )
    parser.add_argument(
        "--task-name-col", default="tabarena_task_name", help="Readable-name column to rewrite."
    )
    args = parser.parse_args()

    migrate_user_task_metadata_csv(
        src_csv=args.src_csv,
        dst_csv=args.src_csv if args.in_place else args.dst_csv,
        task_id_str_col=args.task_id_str_col,
        task_name_col=args.task_name_col,
    )
