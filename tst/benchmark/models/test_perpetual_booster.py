from __future__ import annotations

import pytest


def test_perpetual():
    from autogluon.tabular.testing import FitHelper

    model_hyperparameters = {"iteration_limit": 10, "budget": 0.1}

    try:
        from tabarena.benchmark.models.ag.perpetual_booster.perpetual_booster_model import (
            PerpetualBoosterModel,
        )
        model_cls = PerpetualBoosterModel
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