import pytest


def test_tabm():
    toy_model_params = {"n_epochs": 10, "tabm_k": 2, "n_bins": 8, "num_emb_type": 'none'}
    model_hyperparameters = toy_model_params

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.models.tabm.model import TabMModel
        model_cls = TabMModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
