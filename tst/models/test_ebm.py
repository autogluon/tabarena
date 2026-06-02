from __future__ import annotations

import pytest


def test_ebm():
    model_hyperparameters = {}

    try:
        from autogluon.tabular.models import EBMModel
        from autogluon.tabular.testing import FitHelper

        model_cls = EBMModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}",
        )
