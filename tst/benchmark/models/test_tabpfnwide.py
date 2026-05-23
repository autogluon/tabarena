from __future__ import annotations

import pytest


def test_tabpfnwide():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabpfnwide.tabpfnwide_model import (
            TabPFNWideModel,
        )

        model_cls = TabPFNWideModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={"device": "cpu"})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
