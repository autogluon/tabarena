from __future__ import annotations

from pathlib import Path

from tabarena.evaluation.entrants import ENTRANT_POOLS
from tabarena.evaluation.subset_grid import (
    DATASET_SUBSET_VALUES,
    LITE_VALUES,
    PROBLEM_TYPE_VALUES,
    USE_IMPUTATION_VALUES,
    get_all_subset_combinations,
    get_website_folder_name,
)


def test_grid_is_the_full_product_of_its_axes():
    expected = (
        len(ENTRANT_POOLS)
        * len(USE_IMPUTATION_VALUES)
        * len(PROBLEM_TYPE_VALUES)
        * len(DATASET_SUBSET_VALUES)
        * len(LITE_VALUES)
    )
    combinations = get_all_subset_combinations()
    assert len(combinations) == expected
    assert len(set(combinations)) == expected  # no duplicate cells


def test_every_pool_gets_a_full_subgrid():
    pools = [combination[0] for combination in get_all_subset_combinations()]
    per_pool = {pool.key: pools.count(pool.key) for pool in ENTRANT_POOLS}
    assert len(set(per_pool.values())) == 1, per_pool


def test_website_folder_name_layout():
    """The path *is* the subset's identity, and `Subset.rel_path` in the leaderboard Space
    mirrors it segment for segment. Changing the layout means changing both.
    """
    assert get_website_folder_name(
        entrant_pool="models",
        use_imputation=True,
        problem_type="all",
        dataset_subset=None,
        lite=False,
    ) == Path("website_data/entrants_models/imputation_yes/splits_all/tasks_all/datasets_all")

    assert get_website_folder_name(
        entrant_pool="open_llm_api",
        use_imputation=False,
        problem_type="regression",
        dataset_subset="small",
        lite=True,
    ) == Path("website_data/entrants_open_llm_api/imputation_no/splits_lite/tasks_regression/datasets_small")


def test_every_grid_cell_maps_to_a_distinct_folder():
    folders = {
        get_website_folder_name(
            entrant_pool=entrant_pool,
            use_imputation=use_imputation,
            problem_type=problem_type,
            dataset_subset=dataset_subset,
            lite=lite,
        )
        for entrant_pool, use_imputation, problem_type, _, dataset_subset, lite, _ in get_all_subset_combinations()
    }
    assert len(folders) == len(get_all_subset_combinations())
