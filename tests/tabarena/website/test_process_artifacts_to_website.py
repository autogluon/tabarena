from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from tabarena.website.process_artifacts_to_website import leakage_map, write_n_datasets_index


def test_leakage_map_reads_the_flagged_methods_from_the_website_table():
    board = pd.DataFrame(
        {
            "Model": ["[LimiX-2 (default)](https://arxiv.org/abs/2609.17488)", "TabM (tuned)"],
            "Leakage": ["https://github.com/autogluon/tabarena/issues/648", float("nan")],
            "LeakageDatasets": ["blood-transfusion-service-center;diabetes", float("nan")],
        }
    )
    assert leakage_map(board) == {"LimiX-2": ["blood-transfusion-service-center", "diabetes"]}


def test_a_table_written_before_the_columns_existed_flags_nothing():
    assert leakage_map(pd.DataFrame({"Model": ["TabM (tuned)"]})) == {}


def test_the_dataset_count_index_is_keyed_by_the_cell_folder(tmp_path):
    write_n_datasets_index(tmp_path, {Path("entrants_models/imputation_yes"): 51, Path("subsets/split_all"): 142})
    assert json.loads((tmp_path / "n_datasets.json").read_text()) == {
        "entrants_models/imputation_yes": 51,
        "subsets/split_all": 142,
    }
