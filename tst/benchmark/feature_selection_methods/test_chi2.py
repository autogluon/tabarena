import pytest
from tst.test_feature_selection_method import verify_method

def test_chi2():
    from tabarena.benchmark.feature_selection_methods.ag.chi2.Chi2 import Chi2
    hyperparameters = {"time_limit": 3600, "n_max_features": 10}
    try:
        verify_method(Chi2, hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
