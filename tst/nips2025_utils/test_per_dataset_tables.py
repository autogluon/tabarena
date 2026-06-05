from __future__ import annotations

import pandas as pd

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.per_dataset_tables import _build_dataset_name_map


def _legacy_df() -> pd.DataFrame:
    """A complete legacy frame for dataset d1 (all columns from_legacy_df needs)."""
    return pd.DataFrame(
        {
            "dataset": ["d1"],
            "name": ["d1"],
            "tid": [0],
            "problem_type": ["binary"],
            "n_folds": [1],
            "n_repeats": [1],
            "n_features": [3],
            "n_classes": [2],
            "NumberOfInstances": [10],
            "n_samples_train_per_fold": [6],
            "n_samples_test_per_fold": [4],
            "target_feature": ["t"],
        }
    )


class TestBuildDatasetNameMap:
    def test_none_returns_empty(self):
        assert _build_dataset_name_map(None) == {}

    def test_collection_maps_task_name_to_dataset_name(self):
        # Native: tabarena_task_name (the df_results "dataset" key) -> dataset_name.
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())
        coll.tasks[0].dataset_name = "Pretty Name"  # distinct from tabarena_task_name "d1"
        assert _build_dataset_name_map(coll) == {"d1": "Pretty Name"}

    def test_legacy_df_prefers_dataset_name(self):
        df = _legacy_df()
        df["dataset_name"] = ["Pretty Name"]
        assert _build_dataset_name_map(df) == {"d1": "Pretty Name"}

    def test_legacy_df_falls_back_to_name(self):
        # No dataset_name column -> uses name.
        assert _build_dataset_name_map(_legacy_df()) == {"d1": "d1"}
