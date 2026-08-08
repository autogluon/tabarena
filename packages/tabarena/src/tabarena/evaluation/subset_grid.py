"""The grid of leaderboard subsets, and the folder layout the website reads them from.

One "subset" is one cell of the grid: an entrant pool crossed with the imputation / splits /
tasks / datasets axes. Each cell gets its own evaluation and its own artifact folder, because
the leaderboard's numbers are relative to whatever is in the cell.

Shared by ``evaluation.eval_all`` (which evaluates each cell) and
``plot.tuning_trajectories.plot_pareto_over_tuning_time`` (which draws one trajectory figure
per cell), so the two cannot disagree about what the grid is.
"""

from __future__ import annotations

from itertools import product
from pathlib import Path

from tabarena.evaluation.entrants import ENTRANT_POOLS

#: Values of each axis, in the order the website displays them. First value is the default.
ENTRANT_POOL_KEYS = [pool.key for pool in ENTRANT_POOLS]
USE_IMPUTATION_VALUES = [False, True]
PROBLEM_TYPE_VALUES = ["all", "classification", "regression", "binary", "multiclass"]
DATASET_SUBSET_VALUES = [None, "small", "medium"]
LITE_VALUES = [False, True]


def get_all_subset_combinations() -> list[tuple[str, bool, str, bool, str | None, bool, bool]]:
    """Every cell of the grid, as ``(entrant_pool, use_imputation, problem_type,
    with_baselines, dataset_subset, lite, average_seeds)`` tuples.
    """
    with_baselines_lst = [True]
    average_seeds_lst = [False]

    return list(
        product(
            ENTRANT_POOL_KEYS,
            USE_IMPUTATION_VALUES,
            PROBLEM_TYPE_VALUES,
            with_baselines_lst,
            DATASET_SUBSET_VALUES,
            LITE_VALUES,
            average_seeds_lst,
        )
    )


def get_website_folder_name(
    *,
    entrant_pool: str,
    use_imputation: bool,
    problem_type: str,
    dataset_subset: str | None,
    lite: bool,
) -> Path:
    """The artifact folder for one cell, relative to the generated-artifacts root.

    Mirrored by ``Subset.rel_path`` in the leaderboard Space's ``data_loading.py``: the path
    *is* the subset's identity on both sides.
    """
    folder_name = Path("website_data")
    folder_name = folder_name / f"entrants_{entrant_pool}"
    folder_name = folder_name / ("imputation_yes" if use_imputation else "imputation_no")
    folder_name = folder_name / ("splits_lite" if lite else "splits_all")
    folder_name = folder_name / f"tasks_{problem_type}"
    dataset_subset_name = dataset_subset if dataset_subset is not None else "all"
    return folder_name / f"datasets_{dataset_subset_name}"
