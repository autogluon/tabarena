from __future__ import annotations

import pytest


def test_tabdpt():
    model_hyperparameters = {}

    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.models.tabdpt.model import TabDPTModel

        model_cls = TabDPTModel
        FitHelper.verify_model(
            model_cls=model_cls, model_hyperparameters=model_hyperparameters
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
