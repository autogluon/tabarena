import pytest


def test_realmlp():
    toy_model_params = {"n_epochs": 10}
    model_hyperparameters = toy_model_params

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.models.realmlp.model import RealMLPModel
        model_cls = RealMLPModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
