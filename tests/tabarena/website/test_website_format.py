from __future__ import annotations

import pytest

from tabarena.website.website_format import Constants, get_model_family


@pytest.mark.parametrize(
    ("model_name", "family"),
    [
        # Raw config-type keys.
        ("GBM", Constants.tree),
        ("TA-TABSWIFT", Constants.foundational),
        ("TA-ORION-MSP", Constants.foundational),
        ("MNCA", Constants.neural_network),
        # Display names (used by the Pareto/trajectory plotting paths).
        ("LightGBM", Constants.tree),
        ("RandomForest", Constants.tree),
        ("ExtraTrees", Constants.tree),
        ("PerpetualBooster", Constants.tree),
        ("ModernNCA", Constants.neural_network),
        ("TorchMLP", Constants.neural_network),
        ("OrionMSP", Constants.foundational),
        ("RealTabPFN-2.5", Constants.foundational),
        ("TabDPT-Turbo", Constants.foundational),
        ("iLTM", Constants.foundational),
        ("Nori-30M", Constants.foundational),
        ("Linear", Constants.baseline),
        ("AutoGluon 1.5 (extreme, 4h)", Constants.reference),
        ("xRFM", Constants.other),
    ],
)
def test_get_model_family(model_name: str, family: str):
    assert get_model_family(model_name) == family
