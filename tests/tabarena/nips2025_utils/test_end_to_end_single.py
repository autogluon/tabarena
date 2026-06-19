from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from tabarena.benchmark.task.metadata import TaskMetadataCollection
from tabarena.nips2025_utils.end_to_end_single import (
    EndToEndSingle,
    _filter_file_paths_by_task_metadata,
    _reject_legacy_task_metadata,
)


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
    def test_keeps_matching_tid_drops_others(self):
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())  # tid 7, slug "d1"
        kept = _filter_file_paths_by_task_metadata(_grouped_paths("7", "999"), coll)
        assert set(kept) == {"7/0"}

    def test_keeps_slug_task_dir(self):
        # User/local tasks use the tabarena_task_name slug as the directory, not the int tid.
        coll = TaskMetadataCollection.from_legacy_df(_legacy_df())  # slug "d1"
        kept = _filter_file_paths_by_task_metadata(_grouped_paths("d1", "other_slug"), coll)
        assert set(kept) == {"d1/0"}


class TestRejectLegacyTaskMetadata:
    def test_dataframe_rejected(self):
        with pytest.raises(TypeError, match="no longer accept a legacy"):
            _reject_legacy_task_metadata(_legacy_df())

    def test_none_and_collection_ok(self):
        _reject_legacy_task_metadata(None)  # auto-infer sentinel
        _reject_legacy_task_metadata(TaskMetadataCollection.from_legacy_df(_legacy_df()))


class TestFetchTaskMetadata:
    def test_returns_collection_from_cache(self):
        # tids=[] -> no missing tids -> uses the committed cached metadata (offline), wrapped.
        coll = EndToEndSingle.fetch_task_metadata(tids=[], verbose=False)
        assert isinstance(coll, TaskMetadataCollection)
        assert len(coll) > 0
