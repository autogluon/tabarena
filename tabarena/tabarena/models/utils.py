from __future__ import annotations

import numpy as np


def convert_numpy_dtypes(data: dict) -> dict:
    """Converts NumPy dtypes in a dictionary to Python dtypes.
    Some hyperparameter search space's generate configs with
    numpy dtypes which aren't serializable to yaml. This fixes that.
    """
    converted_data = {}
    for key, value in data.items():
        if isinstance(value, np.generic):
            converted_data[key] = value.item()
        elif isinstance(value, dict):
            converted_data[key] = convert_numpy_dtypes(value)
        elif isinstance(value, list):
            converted_data[key] = [
                convert_numpy_dtypes({i: v})[i] if isinstance(v, (dict, np.generic)) else v for i, v in enumerate(value)
            ]
        else:
            converted_data[key] = value
    return converted_data


def get_configs_generator_from_name(model_name: str):
    """Map a friendly model name to its search-space generator (`gen_<key>`).

    Each entry points at `tabarena.models.<key>.hpo` — the canonical home for
    search-space generators since the per-model folder migration.
    """
    import importlib

    name_to_import_map = {
        "CatBoost": lambda: importlib.import_module("tabarena.models.catboost.hpo").gen_catboost,
        "EBM": lambda: importlib.import_module("tabarena.models.ebm.hpo").gen_ebm,
        "ExtraTrees": lambda: importlib.import_module("tabarena.models.extra_trees.hpo").gen_extratrees,
        "FastaiMLP": lambda: importlib.import_module("tabarena.models.fastai.hpo").gen_fastai,
        "KNN": lambda: importlib.import_module("tabarena.models.knn.hpo").gen_knn,
        "LightGBM": lambda: importlib.import_module("tabarena.models.lightgbm.hpo").gen_lightgbm,
        "Linear": lambda: importlib.import_module("tabarena.models.lr.hpo").gen_linear,
        "ModernNCA": lambda: importlib.import_module("tabarena.models.modernnca.hpo").gen_modernnca,
        "TorchMLP": lambda: importlib.import_module("tabarena.models.nn_torch.hpo").gen_nn_torch,
        "RandomForest": lambda: importlib.import_module("tabarena.models.random_forest.hpo").gen_randomforest,
        "RealMLP": lambda: importlib.import_module("tabarena.models.realmlp.hpo").gen_realmlp,
        "TabDPT": lambda: importlib.import_module("tabarena.models.tabdpt.hpo").gen_tabdpt,
        "TabICL": lambda: importlib.import_module("tabarena.models.tabicl.hpo").gen_tabicl,
        "TabM": lambda: importlib.import_module("tabarena.models.tabm.hpo").gen_tabm,
        "XGBoost": lambda: importlib.import_module("tabarena.models.xgboost.hpo").gen_xgboost,
        "Mitra": lambda: importlib.import_module("tabarena.models.mitra.hpo").gen_mitra,
        "xRFM": lambda: importlib.import_module("tabarena.models.xrfm.hpo").gen_xrfm,
        "RealTabPFN-v2.5": lambda: importlib.import_module("tabarena.models.tabpfnv2_5.hpo").gen_realtabpfnv25,
        "SAP-RPT-OSS": lambda: importlib.import_module("tabarena.models.sap_rpt_oss.hpo").gen_sap_rpt_oss,
        "TabICLv2": lambda: importlib.import_module("tabarena.models.tabicl.hpo").gen_tabiclv2,
        "PerpetualBooster": lambda: importlib.import_module("tabarena.models.perpetual_booster.hpo").gen_perpetual_booster,
        "TabSTAR": lambda: importlib.import_module("tabarena.models.tabstar.hpo").gen_tabstar,
        "TabPFN-2.6": lambda: importlib.import_module("tabarena.models.tabpfnv2_5.hpo").gen_tabpfnv26,
        "iLTM": lambda: importlib.import_module("tabarena.models.iltm.generate").gen_iltm,
        "OrionMSP": lambda: importlib.import_module("tabarena.models.orionmsp.hpo").gen_orionmsp,
        "LimiX": lambda: importlib.import_module("tabarena.models.limix.hpo").gen_limix,
        "TabPFN-3": lambda: importlib.import_module("tabarena.models.tabpfn_3.hpo").gen_tabpfn_3,
        "TabPFN-Wide": lambda: importlib.import_module("tabarena.models.tabpfnwide.hpo").gen_tabpfnwide,
    }

    if model_name not in name_to_import_map:
        raise ValueError(f"Model name '{model_name}' is not recognized. Options are: {list(name_to_import_map.keys())}")

    return name_to_import_map[model_name]()
