from __future__ import annotations

import pandas as pd
import pytest

from tabarena.website.website_format import (
    Constants,
    add_metadata,
    get_model_family,
    system_display_names,
)


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
        ("xRFM", Constants.other),
    ],
)
def test_get_model_family(model_name: str, family: str):
    assert get_model_family(model_name) == family


# -- systems ------------------------------------------------------------------------------


def test_systems_are_classified_from_the_declared_set_not_the_name():
    """The whole point of the ``system_names`` argument: no name prefix table to maintain.

    "AutoGluon" and "PORTFOLIO" used to be hardcoded prefixes, which meant every new system
    landed in "❓ Other" until someone remembered to add it.
    """
    systems = frozenset({"AutoGluon 1.5 (extreme, 4h)", "SomeBrandNewSystem"})
    assert get_model_family("AutoGluon 1.5 (extreme, 4h)", system_names=systems) == Constants.system
    assert get_model_family("SomeBrandNewSystem", system_names=systems) == Constants.system
    # Undeclared, so it falls through to name classification and lands in Other.
    assert get_model_family("SomeBrandNewSystem") == Constants.other


def test_system_display_names_reads_the_metadata_frame():
    info = pd.DataFrame(
        [
            {"display_name": "TabM", "method_class": "model"},
            {"display_name": "AutoGluon 1.5 (extreme, 4h)", "method_class": "system"},
            {"display_name": "TabFM+", "method_class": "system"},
        ]
    )
    assert system_display_names(info) == frozenset({"AutoGluon 1.5 (extreme, 4h)", "TabFM+"})


@pytest.mark.parametrize("info", [None, pd.DataFrame([{"display_name": "TabM"}])])
def test_system_display_names_without_the_column_is_empty(info):
    # An older collection has no method_class column; nothing is a system then.
    assert system_display_names(info) == frozenset()


def _metadata_frame(**overrides) -> pd.DataFrame:
    row = {
        "method": "M",
        "config_type": None,
        "method_type": "baseline",
        "method_subtype": None,
        "display_name": "My System",
        "verified": True,
        "compute": "gpu",
        "reference_url": None,
        "method_class": "system",
        "tags": ("closed-source-api", "with-llm"),
    }
    row.update(overrides)
    return pd.DataFrame([row]).set_index("method")


def test_add_metadata_types_a_system_and_emits_its_tags():
    out = add_metadata(pd.Series({"method": "M"}), metadata_df=_metadata_frame())
    assert out["TypeName"] == Constants.system
    assert out["MethodClass"] == "system"
    assert out["Tags"] == "closed-source-api;with-llm"


def test_add_metadata_leaves_models_to_name_classification():
    out = add_metadata(
        pd.Series({"method": "M"}),
        metadata_df=_metadata_frame(config_type="GBM", method_type="config", method_class="model", tags=()),
    )
    assert out["TypeName"] == Constants.tree
    assert out["MethodClass"] == "model"
    assert out["Tags"] == ""


def test_add_metadata_returns_the_same_keys_when_the_method_is_missing():
    """The caller assigns the result to a fixed column list, so both branches must agree."""
    metadata = _metadata_frame()
    present = add_metadata(pd.Series({"method": "M"}), metadata_df=metadata)
    missing = add_metadata(pd.Series({"method": "Unknown"}), metadata_df=metadata)
    assert set(present.index) == set(missing.index)
