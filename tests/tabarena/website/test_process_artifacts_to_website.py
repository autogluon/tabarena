from __future__ import annotations

import pandas as pd

from tabarena.website.process_artifacts_to_website import leakage_map


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
