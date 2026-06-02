from __future__ import annotations

import pytest


def test_modernnca():
    toy_model_params = {"n_epochs": 10}
    model_hyperparameters = toy_model_params

    try:
        from autogluon.tabular.testing import FitHelper

        from tabarena.models.modernnca.model import ModernNCAModel

        model_cls = ModernNCAModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}",
        )
