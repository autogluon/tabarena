from __future__ import annotations

from typing import TYPE_CHECKING

from numpy.testing import assert_array_equal

from tabarena.benchmark.result import AGBagResult
from tabarena.benchmark.task.openml import OpenMLTaskWrapper
from tabarena.nips2025_utils.artifacts import tabarena_method_metadata_collection

if TYPE_CHECKING:
    import numpy as np

"""
TabArena → OpenML Split Verification Script
===========================================

Purpose
-------
This utility loads precomputed TabArena artifacts for a single method and verifies
that the recorded train/validation/test split indices match the canonical splits
returned by `OpenMLTaskWrapper`. It is intended as a quick integrity check that
TabArena's cached runs (fold/repeat/sample) align with OpenML's official split
definition for each task.

What it does
------------
1. Lists available TabArena methods (so you can pick one to test).
2. Selects one method (default: "Mitra_GPU").
3. Ensures raw artifacts for that method are available locally (downloads if needed).
4. Iterates through each `AGBagResult` in the artifacts and for each run:
   - Prints key metadata (framework, dataset, fold/repeat/sample).
   - Fetches the OpenML split indices for the same (fold, repeat, sample).
   - Asserts:
       * problem type and label are consistent
       * train/val/test indices match
   - Demonstrates how to access per-child validation/test predictions.

Outputs / Side Effects
----------------------
- May download raw artifacts to the local TabArena cache if not present.
- Prints a Markdown table of available methods and per-run metadata lines.

Glossary
--------
- **fold/repeat/sample**: OpenML's CV scheme parameters.
- **child**: A child model inside an AutoGluon bag/ensemble; predictions can be
  examined per child via the provided accessors.

"""


if __name__ == "__main__":
    # 1) Surface available methods so users can pick a target method quickly.
    methods_info = tabarena_method_metadata_collection.info()
    print(methods_info.to_markdown())

    # Choose a method to validate. Mitra has a single config, which keeps this fast.
    method = "Mitra_GPU"
    method_metadata = tabarena_method_metadata_collection.get_method_metadata(method=method)

    # 2) Ensure raw artifacts exist locally (download once, then cached).
    if not method_metadata.path_raw_exists:
        method_metadata.method_downloader().download_raw()

    # 3) Load all run results for the chosen method.
    results_lst: list[AGBagResult] = method_metadata.load_raw()

    # Cache of OpenML tasks by id (avoid re-fetching for each run).
    tasks: dict[int, OpenMLTaskWrapper] = {}

    # 4) Loop over each run and verify splits + metadata.
    for task_result in results_lst:
        assert isinstance(task_result, AGBagResult)

        # Helpful progress line for humans skimming the output.
        fold, repeat, sample = task_result.fold, task_result.repeat, task_result.sample
        print(
            f"framework={task_result.framework}, "
            f"dataset={task_result.dataset}, "
            f"fold={fold}, "
            f"repeat={repeat}, "
            f"sample={sample}",
        )

        # OpenML task id associated with this run's dataset/splits.
        task_id: int = task_result.task_metadata["tid"]

        # Lazily construct/fetch the OpenML task wrapper (memoized in `tasks`).
        if (task := tasks.get(task_id)) is None:
            task = tasks[task_id] = OpenMLTaskWrapper.from_task_id(task_id=task_id)

        # Obtain canonical OpenML split indices for the same CV coordinates.
        train_indices, test_indices = task.get_split_indices(fold=fold, repeat=repeat, sample=sample)

        # Basic metadata consistency checks (problem type & label/target).
        assert task.problem_type == task_result.problem_type, (
            f"Problem type mismatch for task_id={task_id}: {task.problem_type=} vs {task_result.problem_type=}"
        )
        assert task.label == task_result.label, (
            f"Label/target mismatch for task_id={task_id}: {task.label=} vs {task_result.label=}"
        )

        # Split alignment checks: TabArena recorded indices must match OpenML's reference.
        assert_array_equal(train_indices, task_result.y_val_idx)
        assert_array_equal(test_indices, task_result.y_test_idx)

        # predictions in numpy format
        y_pred_proba_val: np.ndarray = task_result.y_pred_proba_val
        y_pred_proba_test: np.ndarray = task_result.y_pred_proba_test

        # predictions in pandas format with correct idx and column names
        y_pred_proba_val_as_pd = task_result.y_pred_proba_val_as_pd
        y_pred_proba_test_as_pd = task_result.y_pred_proba_test_as_pd

        # If the run is a bag/ensemble, you can also inspect per-child predictions.
        num_children = task_result.num_children
        for child_idx in range(num_children):
            # Child-specific validation indices (subset of train/val area).
            child_val_indices = task_result.val_idx_child(child_idx)

            # Predictions as pandas DataFrame/Series for validation/test splits.
            # These can be fed into ensembling/stacking diagnostics or calibration checks.
            y_pred_proba_val_child_as_pd = task_result.y_pred_proba_val_child_as_pd(idx=child_idx)
            y_pred_proba_test_child_as_pd = task_result.y_pred_proba_test_child_as_pd(idx=child_idx)
