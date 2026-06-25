"""Generate the BeyondArena leaderboard and (re)generate the committed "core" subset.

The ``"core"`` subset keeps each dataset's first ``folds_to_use`` splits, where
``folds_to_use = min(folds_needed_for_stability, num_folds)`` comes from the fold-similarity
analysis (``compare(compute_fold_similarity=True)``). Running this script rewrites the committed
``BeyondArena_core_tasks.csv`` that the ``BeyondArenaContext`` ``"core"`` subset predicate reads,
then shows the leaderboard restricted to that core via ``subset=["core"]``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.contexts.beyondarena.context import CORE_TASKS_CSV, BeyondArenaContext

TARGET_RELIABILITY = 0.8

if __name__ == "__main__":
    save_path = Path("output_beyondarena_leaderboard")  # folder to save all figures and tables

    tabarena_context = BeyondArenaContext()

    # 1) full leaderboard + fold-similarity analysis (writes fold_similarity.csv).
    leaderboard = tabarena_context.compare(
        output_dir=save_path,
        compute_fold_similarity=True,
        fold_similarity_kwargs={"target_reliability": TARGET_RELIABILITY},
    )
    print("Leaderboard:")
    print(tabarena_context.leaderboard_to_website_format(leaderboard=leaderboard).to_markdown(index=False))
    print("")

    # 2) folds_to_use = min(folds_needed_for_stability, num_folds), per dataset. The `min` caps the
    #    stability estimate at the folds actually available; datasets whose stability could not be
    #    estimated (NaN folds_needed) fall back to num_folds via min(skipna).
    fold_similarity = pd.read_csv(save_path / "fold_similarity.csv")
    folds_needed_col = f"folds_needed_for_stability@{TARGET_RELIABILITY}"
    fold_similarity["folds_to_use"] = fold_similarity[[folds_needed_col, "num_folds"]].min(axis=1)
    folds_to_use = {
        str(d): int(n) for d, n in zip(fold_similarity["dataset"], fold_similarity["folds_to_use"], strict=False)
    }

    # 3) expand to valid tasks: each dataset's first `folds_to_use` splits (lowest split indices).
    grid = tabarena_context.task_metadata_collection.task_grid()
    rows = []
    for dataset, dataset_grid in grid.groupby("dataset"):
        splits = sorted(int(s) for s in dataset_grid["split"].unique())
        n = min(folds_to_use.get(str(dataset), len(splits)), len(splits))
        rows.extend({"dataset": dataset, "split": s} for s in splits[:n])
    valid_tasks = pd.DataFrame(rows).sort_values(["dataset", "split"]).reset_index(drop=True)

    # 4) commit the valid tasks; the BeyondArena "core" subset predicate reads this file.
    valid_tasks.to_csv(CORE_TASKS_CSV, index=False)
    print(f"Wrote {len(valid_tasks)} core (dataset, split) tasks (of {len(grid)}) to {CORE_TASKS_CSV}\n")

    # 5) the committed "core" subset now restricts the leaderboard to those tasks.
    core_leaderboard = tabarena_context.compare(output_dir=save_path / "core", subset=["core"])
    print("Core leaderboard:")
    print(tabarena_context.leaderboard_to_website_format(leaderboard=core_leaderboard).to_markdown(index=False))
