"""`AGWrapperV2` fold-count resolution and structure declaration, without fitting anything."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from autogluon.core.metrics import get_metric

from tabarena.benchmark.exec_models.autogluon_v2 import AGWrapperV2
from tabarena.benchmark.task.metadata import AUTO_NUM_SPLITS, GroupLabelTypes, ValidationMetadata


def _wrapper(metadata: ValidationMetadata, *, use_task_specific_validation: bool) -> AGWrapperV2:
    return AGWrapperV2(
        problem_type="binary",
        eval_metric=get_metric("roc_auc", problem_type="binary"),
        validation_metadata=metadata,
        use_task_specific_validation=use_task_specific_validation,
    )


def _grouped_data(n_rows: int, n_groups: int) -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"feature": np.arange(n_rows, dtype=float), "grp": [f"g{i % n_groups}" for i in range(n_rows)]})
    return X, pd.Series(np.arange(n_rows) % 2)


def test_auto_counts_on_a_tiny_grouped_task_take_the_protocol_and_declare_the_structure():
    metadata = ValidationMetadata(group_on="grp", group_labels=GroupLabelTypes.PER_SAMPLE)
    X, y = _grouped_data(n_rows=400, n_groups=20)
    fit_kwargs = {"num_bag_folds": AUTO_NUM_SPLITS, "num_bag_sets": AUTO_NUM_SPLITS}
    num_folds = _wrapper(metadata, use_task_specific_validation=True)._apply_validation_splits(fit_kwargs, X=X, y=y)
    assert num_folds == ValidationMetadata.tiny_data_num_folds
    assert fit_kwargs["num_bag_folds"] == ValidationMetadata.tiny_data_num_folds
    assert fit_kwargs["num_bag_sets"] == ValidationMetadata.tiny_data_num_repeats
    assert fit_kwargs["validation_structure"].group_on == "grp"


def test_explicit_counts_on_a_tiny_grouped_task_are_kept():
    metadata = ValidationMetadata(group_on="grp", group_labels=GroupLabelTypes.PER_SAMPLE)
    X, y = _grouped_data(n_rows=400, n_groups=20)
    fit_kwargs = {"num_bag_folds": 2, "num_bag_sets": 2}
    num_folds = _wrapper(metadata, use_task_specific_validation=True)._apply_validation_splits(fit_kwargs, X=X, y=y)
    assert num_folds == 2
    assert fit_kwargs["num_bag_folds"] == 2
    assert fit_kwargs["num_bag_sets"] == 2
    assert fit_kwargs["validation_structure"].group_on == "grp"


def test_auto_counts_without_task_specific_validation_are_the_defaults():
    """The data size must not decide here, so a tiny task still gets the defaults and no structure."""
    metadata = ValidationMetadata(group_on="grp", group_labels=GroupLabelTypes.PER_SAMPLE)
    X, y = _grouped_data(n_rows=400, n_groups=20)
    fit_kwargs = {"num_bag_folds": AUTO_NUM_SPLITS, "num_bag_sets": AUTO_NUM_SPLITS}
    num_folds = _wrapper(metadata, use_task_specific_validation=False)._apply_validation_splits(fit_kwargs, X=X, y=y)
    assert num_folds == ValidationMetadata.default_num_folds
    assert fit_kwargs["num_bag_folds"] == ValidationMetadata.default_num_folds
    assert fit_kwargs["num_bag_sets"] == ValidationMetadata.default_num_repeats
    assert "validation_structure" not in fit_kwargs


def test_unstructured_task_declares_no_structure_but_still_resolves_counts():
    X, y = _grouped_data(n_rows=400, n_groups=20)
    fit_kwargs = {"num_bag_folds": AUTO_NUM_SPLITS, "num_bag_sets": AUTO_NUM_SPLITS}
    _wrapper(ValidationMetadata(), use_task_specific_validation=True)._apply_validation_splits(fit_kwargs, X=X, y=y)
    assert fit_kwargs["num_bag_folds"] == ValidationMetadata.tiny_data_num_folds
    assert "validation_structure" not in fit_kwargs


@pytest.mark.parametrize("use_task_specific_validation", [True, False])
def test_no_fold_count_is_left_alone(use_task_specific_validation):
    X, y = _grouped_data(n_rows=400, n_groups=20)
    fit_kwargs: dict = {}
    num_folds = _wrapper(
        ValidationMetadata(), use_task_specific_validation=use_task_specific_validation
    )._apply_validation_splits(fit_kwargs, X=X, y=y)
    assert num_folds is None
    assert "num_bag_folds" not in fit_kwargs
