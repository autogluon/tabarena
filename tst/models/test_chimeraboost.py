from __future__ import annotations

import pytest


def test_chimeraboost():
    # small tree budget to keep the toy-dataset test snappy (default cap is 10k)
    model_hyperparameters = {"n_estimators": 100}

    try:
        from autogluon.tabular.testing import FitHelper

        from tabarena.models.chimeraboost.model import ChimeraBoostModel

        model_cls = ChimeraBoostModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters=model_hyperparameters)
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}",
        )
