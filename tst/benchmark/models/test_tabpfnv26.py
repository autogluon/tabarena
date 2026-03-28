from __future__ import annotations

import pytest


def test_tabpfn26():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.tabpfnv2_5.tabpfnv2_5_model import (
            TabPFNv26Model,
        )

        model_cls = TabPFNv26Model
        FitHelper.verify_model(
            model_cls=model_cls,
            model_hyperparameters={
                "n_estimators": 1,
            },
        )
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
