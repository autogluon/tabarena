from __future__ import annotations

from pathlib import Path

import pytest

from tabarena.evaluation.entrants import ENTRANT_POOLS
from tabarena.evaluation.subset_grid import get_all_subset_combinations
from tabarena.website.build import cell_label, website_cells


def test_twin_pools_are_computed_once_and_published_under_every_key():
    groups = [[pool.key for pool in ENTRANT_POOLS[i : i + 2]] for i in range(0, len(ENTRANT_POOLS), 2)]
    cells = website_cells(groups)
    assert len(cells) == len(get_all_subset_combinations()) // 2
    assert {cell.entrant_pool for cell in cells} == {group[0] for group in groups}
    folders = [folder for cell in cells for folder in cell.folders]
    assert len(folders) == len(set(folders)) == len(get_all_subset_combinations())
    first = cells[0]
    assert first.folders == tuple(
        f"entrants_{key}/imputation_no/splits_all/tasks_all/datasets_all" for key in groups[0]
    )


def test_a_cell_lists_its_task_filters_in_grid_order():
    cell = next(
        c
        for c in website_cells([[pool.key] for pool in ENTRANT_POOLS])
        if c.problem_type == "binary" and c.dataset_subset == "small" and c.lite
    )
    assert cell.subset == ["binary", "small", "lite"]


def test_cell_label_names_what_differs_from_the_defaults():
    label = cell_label(Path("entrants_models/imputation_yes/splits_all/tasks_all/datasets_small"))
    assert label == f"{ENTRANT_POOLS[0].label} | All Tasks | Small"
    with pytest.raises(KeyError, match="No label for entrant pool"):
        cell_label(Path("entrants_nobody/imputation_yes"))
