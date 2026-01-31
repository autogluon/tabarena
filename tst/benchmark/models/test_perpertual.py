from __future__ import annotations

import pytest


def test_perpetual():
    from autogluon.tabular.testing import FitHelper

    model_hyperparameters = {"iteration_limit": 10, "budget": 0.1}

    try:
        from tabarena.benchmark.models.ag.perpetual.perpetual_model import (
            PerpetualBoostingModel,
        )
        model_cls = PerpetualBoostingModel
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters=model_hyperparameters,
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
