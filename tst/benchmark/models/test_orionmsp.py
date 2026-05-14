from __future__ import annotations

import pytest


def test_orionmsp():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.orionmsp.orionmsp_model import (
            OrionMSPModel,
        )

        model_cls = OrionMSPModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={"n_estimators": 1})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
