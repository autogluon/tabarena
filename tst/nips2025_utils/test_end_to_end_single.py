from __future__ import annotations

from pathlib import Path

import pandas as pd

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.end_to_end_single import _filter_file_paths_by_task_metadata


def _legacy_df() -> pd.DataFrame:
    """A complete legacy frame for dataset d1 / tid 7 (all columns from_legacy_df needs)."""
    return pd.DataFrame(
        {
            "dataset": ["d1"],
            "name": ["d1"],
            "tid": [7],
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


def _grouped_paths(*task_dirs: str) -> dict[str, list[Path]]:
    # Keys are "{task_dir}/{split}" as produced by the real grouping logic.
    return {f"{d}/0": [Path(f"/raw/{d}/0/results.pkl")] for d in task_dirs}


class TestFilterFilePathsByTaskMetadata:
    def test_collection_keeps_matching_tid_drops_others(self):
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())  # tid 7, slug "d1"
        grouped = _grouped_paths("7", "999")  # 7 in-suite, 999 not
        kept = _filter_file_paths_by_task_metadata(grouped, coll)
        assert set(kept) == {"7/0"}

    def test_collection_keeps_slug_task_dir(self):
        # User/local tasks use the tabarena_task_name slug as the directory, not the int tid.
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())  # slug "d1"
        grouped = _grouped_paths("d1", "other_slug")
        kept = _filter_file_paths_by_task_metadata(grouped, coll)
        assert set(kept) == {"d1/0"}

    def test_collection_matches_legacy_df_behavior(self):
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())
        grouped = _grouped_paths("7", "d1", "999")
        kept_coll = _filter_file_paths_by_task_metadata(grouped, coll)
        kept_df = _filter_file_paths_by_task_metadata(grouped, coll.to_legacy_df())
        assert set(kept_coll) == set(kept_df) == {"7/0", "d1/0"}
