import pytest
from tst.test_feature_selection_method import verify_method

@pytest.mark.parametrize("problem_type, dataset_id", [
    ("binary", 55),
    ("multiclass", 10),
    ("regression", 46964),
])
def test_disr(problem_type, dataset_id):
    from tabarena.benchmark.feature_selection_methods.ag.disr.DISR import DISR
    hyperparameters = {"n_max_features": 10, "time_limit": 3600, "dataset_id": dataset_id, "problem_type": problem_type}
    try:
        verify_method(DISR, hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
