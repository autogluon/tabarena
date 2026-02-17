from __future__ import annotations

import pytest


def test_grande():
    from autogluon.tabular.testing import FitHelper

    model_hyperparameters = {"n_estimators": 2, "depth": 3, "num_emb_n_bins": 2}

    try:
        from tabarena.benchmark.models.ag import GRANDEModel

        model_cls = GRANDEModel
        FitHelper.verify_model(
            model_cls=model_cls, model_hyperparameters=model_hyperparameters
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
