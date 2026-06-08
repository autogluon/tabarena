"""Compare the two TabArena ``task_metadata`` loading methods side by side.

There are currently two ways to obtain task metadata for the TabArena v0.1 suite:

1. **Legacy** — ``load_task_metadata(paper=True)`` reads the committed
   ``task_metadata_tabarena51.csv`` and returns a one-row-per-dataset DataFrame
   (the format ``TabArenaContext`` / ``ExperimentBatchRunner`` consume directly).

2. **New** — a ``TabArenaMetadataBundle`` (here ``TabArenaV0pt1MetadataBundle``)
   loads a ``list[TabArenaTaskMetadata]``, which we then convert to the legacy
   DataFrame via :func:`tabarena.benchmark.task.metadata.to_legacy_task_metadata`.

This script loads both, then reports: shapes, shared/diverging columns, dataset
overlap, and per-column value agreement on the columns the downstream consumers read.

Run:
    python examples/meta/compare_task_metadata_sources.py
"""

from __future__ import annotations

import pandas as pd

from tabarena.benchmark.task.metadata import (
    TabArenaV0pt1MetadataBundle,
    to_legacy_task_metadata,
)
from tabarena.nips2025_utils.fetch_metadata import load_task_metadata

# Columns the downstream consumers (TabArenaContext, ExperimentBatchRunner, and the
# subset predicates) actually read from task_metadata. `n_samples_train_per_fold` is
# aliased to `max_train_rows` by the subset predicates.
COMPARE_COLUMNS = [
    "tid",
    "n_folds",
    "n_repeats",
    "problem_type",
    "n_features",
    "n_classes",
    "n_samples_train_per_fold",
]

KEY = "dataset"


def _rule(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def load_legacy() -> pd.DataFrame:
    """Method 1: the original CSV-backed loader."""
    return load_task_metadata(paper=True)


def load_via_bundle() -> pd.DataFrame:
    """Method 2: bundle -> list[TabArenaTaskMetadata] -> legacy DataFrame.

    ``materialize=False`` keeps this metadata-only (no OpenML dataset downloads).
    """
    bundle = TabArenaV0pt1MetadataBundle(materialize=False)
    task_metadata = bundle.load_task_metadata()
    return to_legacy_task_metadata(task_metadata)


def compare_columns(old: pd.DataFrame, new: pd.DataFrame) -> None:
    _rule("Shapes & columns")
    print(f"legacy load_task_metadata : {old.shape[0]:>4} rows x {old.shape[1]:>3} cols")
    print(f"bundle + to_legacy        : {new.shape[0]:>4} rows x {new.shape[1]:>3} cols")

    old_cols, new_cols = set(old.columns), set(new.columns)
    shared = sorted(old_cols & new_cols)
    print(f"\nshared columns ({len(shared)}): {shared}")
    print(f"legacy-only columns ({len(old_cols - new_cols)}): {sorted(old_cols - new_cols)}")
    print(f"bundle-only columns ({len(new_cols - old_cols)}): {sorted(new_cols - old_cols)}")

    missing = [c for c in COMPARE_COLUMNS if c not in shared]
    if missing:
        print(f"\n!! comparison columns absent from one side: {missing}")


def compare_datasets(old: pd.DataFrame, new: pd.DataFrame) -> list[str]:
    _rule("Dataset overlap (keyed on 'dataset')")
    old_ds, new_ds = set(old[KEY]), set(new[KEY])
    shared = sorted(old_ds & new_ds)
    print(f"legacy datasets : {len(old_ds)}")
    print(f"bundle datasets : {len(new_ds)}")
    print(f"shared datasets : {len(shared)}")
    if old_ds - new_ds:
        print(f"legacy-only datasets: {sorted(old_ds - new_ds)}")
    if new_ds - old_ds:
        print(f"bundle-only datasets: {sorted(new_ds - old_ds)}")
    return shared


def compare_values(old: pd.DataFrame, new: pd.DataFrame, shared_datasets: list[str]) -> None:
    _rule("Per-column value agreement (on shared datasets)")
    cols = [c for c in COMPARE_COLUMNS if c in old.columns and c in new.columns]
    o = old[old[KEY].isin(shared_datasets)][[KEY, *cols]].set_index(KEY).sort_index()
    n = new[new[KEY].isin(shared_datasets)][[KEY, *cols]].set_index(KEY).sort_index()
    merged = o.join(n, lsuffix="_legacy", rsuffix="_bundle")

    for col in cols:
        left, right = merged[f"{col}_legacy"], merged[f"{col}_bundle"]
        # equal_nan-style comparison that tolerates dtype differences (e.g. int vs float)
        if pd.api.types.is_numeric_dtype(left) and pd.api.types.is_numeric_dtype(right):
            equal = left.fillna(-1).astype(float) == right.fillna(-1).astype(float)
        else:
            equal = left.astype("string").fillna("<NA>") == right.astype("string").fillna("<NA>")
        n_eq = int(equal.sum())
        verdict = "OK " if n_eq == len(merged) else "DIFF"
        print(f"  [{verdict}] {col:<26} {n_eq}/{len(merged)} match")
        if n_eq != len(merged):
            diff = merged.loc[~equal, [f"{col}_legacy", f"{col}_bundle"]]
            with pd.option_context("display.max_rows", 20, "display.width", 120):
                print(diff.to_string())


def main() -> None:
    print("Loading TabArena v0.1 task metadata via both methods...")
    old = load_legacy()
    new = load_via_bundle()

    compare_columns(old, new)
    shared_datasets = compare_datasets(old, new)
    if shared_datasets:
        compare_values(old, new, shared_datasets)

    _rule("Side-by-side head (shared comparison columns)")
    cols = [KEY, *[c for c in COMPARE_COLUMNS if c in old.columns and c in new.columns]]
    with pd.option_context("display.width", 160, "display.max_columns", None):
        print("\n-- legacy --")
        print(old[cols].sort_values(KEY).head().to_string(index=False))
        print("\n-- bundle --")
        print(new[cols].sort_values(KEY).head().to_string(index=False))


if __name__ == "__main__":
    main()
