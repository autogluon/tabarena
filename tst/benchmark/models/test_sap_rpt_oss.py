from __future__ import annotations

import pytest


def test_sap_rpt_oss():
    try:
        from autogluon.tabular.testing import FitHelper
        from tabarena.benchmark.models.ag.sap_rpt_oss.sap_rpt_oss_model import (
            SAPRPTOSSModel,
        )

        model_cls = SAPRPTOSSModel
        FitHelper.verify_model(model_cls=model_cls, model_hyperparameters={})
    except ImportError as err:
        pytest.skip(
            f"Import Error, skipping test... "
            f"Ensure you have the proper dependencies installed to run this test:\n"
            f"{err}"
        )
