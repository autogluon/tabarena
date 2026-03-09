from __future__ import annotations

import pytest

from tst.benchmark.feature_selection_methods.fs_test_utils import verify_method


@pytest.mark.parametrize(
    ("problem_type", "dataset_id"),
    [
        ("binary", 55),
        ("multiclass", 10),
        ("regression", 46964),
    ],
)
def test_lasso(problem_type, dataset_id):
    from tabarena.benchmark.feature_selection_methods.ag.lasso.Lasso import Lasso

    hyperparameters = {"n_max_features": 10, "time_limit": 3600, "dataset_id": dataset_id, "problem_type": problem_type}
    try:
        verify_method(Lasso, hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... Ensure you have the proper dependencies installed to run this test:\n{err}"
        )
