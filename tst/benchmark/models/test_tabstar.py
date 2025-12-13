from __future__ import annotations

import pytest


def test_tabstar():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabstar.tabstar_model import (
            TabStarModel,
        )

        model_cls = TabStarModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={"max_epochs": 1})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
