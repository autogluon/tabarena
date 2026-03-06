import pytest
from tst.test_feature_selection_method import verify_method

def test_relieff():
    from tabarena.benchmark.feature_selection_methods.ag.relieff.ReliefF import ReliefF
    hyperparameters = {"time_limit": 3600, "n_max_features": 10}
    try:
        verify_method(ReliefF, hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
