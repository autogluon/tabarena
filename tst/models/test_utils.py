from __future__ import annotations

import importlib

import pytest

from tabarena.models.utils import get_configs_generator_from_name

# Maps every previously-hard-coded friendly name to the (module, generator)
# pair its old `name_to_import_map` resolved to. The registry-driven lookup
# in `get_configs_generator_from_name` must return *the same object* for
# every entry. Locked in as a regression test so future registry changes
# don't silently re-route a public name.
EXPECTED: dict[str, tuple[str, str]] = {
    "CatBoost": ("tabarena.models.catboost.hpo", "gen_catboost"),
    "EBM": ("tabarena.models.ebm.hpo", "gen_ebm"),
    "ExtraTrees": ("tabarena.models.extra_trees.hpo", "gen_extratrees"),
    "FastaiMLP": ("tabarena.models.fastai.hpo", "gen_fastai"),
    "KNN": ("tabarena.models.knn.hpo", "gen_knn"),
    "LightGBM": ("tabarena.models.lightgbm.hpo", "gen_lightgbm"),
    "Linear": ("tabarena.models.lr.hpo", "gen_linear"),
    "ModernNCA": ("tabarena.models.modernnca.hpo", "gen_modernnca"),
    "TorchMLP": ("tabarena.models.nn_torch.hpo", "gen_nn_torch"),
    "RandomForest": ("tabarena.models.random_forest.hpo", "gen_randomforest"),
    "RealMLP": ("tabarena.models.realmlp.hpo", "gen_realmlp"),
    "TabDPT": ("tabarena.models.tabdpt.hpo", "gen_tabdpt"),
    "TabICL": ("tabarena.models.tabicl.hpo", "gen_tabicl"),
    "TabM": ("tabarena.models.tabm.hpo", "gen_tabm"),
    "XGBoost": ("tabarena.models.xgboost.hpo", "gen_xgboost"),
    "Mitra": ("tabarena.models.mitra.hpo", "gen_mitra"),
    "xRFM": ("tabarena.models.xrfm.hpo", "gen_xrfm"),
    "RealTabPFN-v2.5": ("tabarena.models.tabpfnv2_5.hpo", "gen_realtabpfnv25"),
    "SAP-RPT-OSS": ("tabarena.models.sap_rpt_oss.hpo", "gen_sap_rpt_oss"),
    "TabICLv2": ("tabarena.models.tabicl.hpo", "gen_tabiclv2"),
    "PerpetualBooster": ("tabarena.models.perpetual_booster.hpo", "gen_perpetual_booster"),
    "TabSTAR": ("tabarena.models.tabstar.hpo", "gen_tabstar"),
    "TabPFN-2.6": ("tabarena.models.tabpfnv2_5.hpo", "gen_tabpfnv26"),
    "OrionMSP": ("tabarena.models.orionmsp.hpo", "gen_orionmsp"),
    "LimiX": ("tabarena.models.limix.hpo", "gen_limix"),
    "TabPFN-3": ("tabarena.models.tabpfn_3.hpo", "gen_tabpfn_3"),
    "TabPFN-Wide": ("tabarena.models.tabpfnwide.hpo", "gen_tabpfnwide"),
}


@pytest.mark.parametrize(("friendly", "module_attr"), list(EXPECTED.items()))
def test_get_configs_generator_from_name_preserves_legacy_mapping(friendly, module_attr):
    module_path, attr_name = module_attr
    canonical = getattr(importlib.import_module(module_path), attr_name)
    assert get_configs_generator_from_name(friendly) is canonical


def test_get_configs_generator_from_name_unknown_raises():
    with pytest.raises(ValueError, match="not recognized"):
        get_configs_generator_from_name("NotARealModel")


def test_get_configs_generator_from_name_error_message_lists_options():
    with pytest.raises(ValueError) as excinfo:
        get_configs_generator_from_name("NotARealModel")
    msg = str(excinfo.value)
    # Spot-check that a few known friendly names appear in the suggestions.
    for name in ("CatBoost", "EBM", "TabPFN-3", "TabPFN-Wide"):
        assert name in msg, f"expected {name!r} in error message, got: {msg}"
