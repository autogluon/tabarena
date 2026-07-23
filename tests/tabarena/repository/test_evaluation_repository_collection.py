from __future__ import annotations

import warnings

import pandas as pd
import pytest

from tabarena.repository.evaluation_repository_collection import merge_metadata_frames


class TestMergeMetadataFrames:
    """merge_metadata_frames joins per-dataset metadata on the dataset identity only."""

    def test_disjoint_datasets_with_mismatched_descriptive_dtype_merge(self):
        # Regression test: a descriptive column is object in one frame (a classification
        # dataset) and all-NaN -> float64 in another (a regression dataset). It must NOT be
        # used as a join key (pandas refuses to merge object vs float64 keys); joining on
        # `dataset` only lets the frames merge and coalesces the descriptive column.
        clf = pd.DataFrame(
            {"dataset": ["ds_clf"], "tid": [1], "problem_type": ["binary"], "class_consistency": ["consistent"]},
        )
        reg = pd.DataFrame(
            {"dataset": ["ds_reg"], "tid": [2], "problem_type": ["regression"], "class_consistency": [float("nan")]},
        )
        assert clf["class_consistency"].dtype == object
        assert reg["class_consistency"].dtype == "float64"

        merged = merge_metadata_frames([clf, reg]).set_index("dataset")
        assert sorted(merged.index) == ["ds_clf", "ds_reg"]
        assert merged.loc["ds_clf", "class_consistency"] == "consistent"
        assert pd.isna(merged.loc["ds_reg", "class_consistency"])

    def test_overlapping_dataset_coalesces_nan_with_value(self):
        # Same dataset across frames: NaN on one side, a value on the other -> one row, value kept.
        nan_side = pd.DataFrame({"dataset": ["ds"], "class_consistency": [float("nan")]})
        val_side = pd.DataFrame({"dataset": ["ds"], "class_consistency": ["consistent"]})
        merged = merge_metadata_frames([nan_side, val_side])
        assert len(merged) == 1
        assert merged.loc[0, "class_consistency"] == "consistent"

    def test_does_not_emit_incompatible_dtype_future_warning(self):
        clf = pd.DataFrame({"dataset": ["ds_clf"], "class_consistency": ["consistent"]})
        reg = pd.DataFrame({"dataset": ["ds_reg"], "class_consistency": [float("nan")]})
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            merge_metadata_frames([clf, reg])

    def test_genuine_conflict_still_raises(self):
        a = pd.DataFrame({"dataset": ["ds"], "problem_type": ["binary"]})
        b = pd.DataFrame({"dataset": ["ds"], "problem_type": ["multiclass"]})
        with pytest.raises(ValueError, match="Conflict detected"):
            merge_metadata_frames([a, b])

    def test_no_identity_key_raises(self):
        a = pd.DataFrame({"foo": [1]})
        b = pd.DataFrame({"foo": [2]})
        with pytest.raises(ValueError, match="identity key"):
            merge_metadata_frames([a, b])

    def test_descriptive_conflict_keeps_first_and_warns(self, caplog):
        # Legacy artifacts store integer per-fold sample counts; newer converters store
        # fractional across-fold means. Such conflicts must not fail the merge: the
        # earlier frame's value wins, with a warning.
        import logging

        legacy = pd.DataFrame({"dataset": ["ds"], "n_samples_test_per_fold": [299]})
        new = pd.DataFrame({"dataset": ["ds"], "n_samples_test_per_fold": [299.3333333]})
        with caplog.at_level(logging.WARNING):
            merged = merge_metadata_frames([legacy, new])
        assert len(merged) == 1
        assert merged.loc[0, "n_samples_test_per_fold"] == 299
        assert any("n_samples_test_per_fold" in r.message and "first repo" in r.message for r in caplog.records)

    def test_repurposed_task_type_conflict_keeps_first(self):
        # Newer converters put the split type into `task_type`; the legacy OpenML task
        # type from the earlier frame must win.
        legacy = pd.DataFrame({"dataset": ["ds"], "task_type": ["Supervised Regression"]})
        new = pd.DataFrame({"dataset": ["ds"], "task_type": ["random"]})
        merged = merge_metadata_frames([legacy, new])
        assert merged.loc[0, "task_type"] == "Supervised Regression"

    def test_descriptive_conflict_does_not_mask_strict_conflict(self):
        # A frame pair with both a descriptive conflict and a strict (problem_type)
        # conflict must still raise on the strict column.
        a = pd.DataFrame({"dataset": ["ds"], "problem_type": ["binary"], "n_samples_test_per_fold": [100]})
        b = pd.DataFrame({"dataset": ["ds"], "problem_type": ["regression"], "n_samples_test_per_fold": [100.5]})
        with pytest.raises(ValueError, match="problem_type"):
            merge_metadata_frames([a, b])
