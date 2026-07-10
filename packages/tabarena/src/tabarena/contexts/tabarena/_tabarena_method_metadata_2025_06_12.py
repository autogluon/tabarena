from __future__ import annotations

import copy

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models.catboost.info import catboost_method_metadata
from tabarena.models.extra_trees.info import extra_trees_method_metadata
from tabarena.models.fastai.info import fastai_method_metadata
from tabarena.models.lightgbm.info import lightgbm_method_metadata
from tabarena.models.modernnca.info import (
    modernnca_gpu_method_metadata,
    modernnca_method_metadata,
)
from tabarena.models.nn_torch.info import nn_torch_method_metadata
from tabarena.models.random_forest.info import random_forest_method_metadata
from tabarena.models.realmlp.info import realmlp_cpu_method_metadata
from tabarena.models.tabicl.info import tabicl_method_metadata
from tabarena.models.tabm.info import tabm_gpu_method_metadata, tabm_method_metadata
from tabarena.models.xgboost.info import xgboost_method_metadata

# Models that own their MethodMetadata via per-model `info.py` modules — seeded directly into
# `methods_2025_06_12`. The factory loop below builds only the remaining 2025-06-12 methods that
# have no per-model `info.py` (the maps/lists in this file are therefore keyed by just those).
_per_model_metadata = [
    catboost_method_metadata,
    extra_trees_method_metadata,
    fastai_method_metadata,
    lightgbm_method_metadata,
    modernnca_method_metadata,
    modernnca_gpu_method_metadata,
    nn_torch_method_metadata,
    random_forest_method_metadata,
    realmlp_cpu_method_metadata,
    tabicl_method_metadata,
    tabm_method_metadata,
    tabm_gpu_method_metadata,
    xgboost_method_metadata,
]

methods_2025_06_12 = list(_per_model_metadata)

common_kwargs = dict(
    suite="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="config",
)

cpu_kwargs = dict(
    compute="cpu",
    **common_kwargs,
)

gpu_kwargs = dict(
    compute="gpu",
    name_suffix="_GPU",
    **common_kwargs,
)

# The 2025-06-12 methods that have no per-model `info.py`; built from the maps below.
methods = [
    "Dummy",
    "ExplainableBM",
    "KNeighbors",
    "LinearModel",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabPFNv2_GPU",
]

# Methods that should not be tuned/tuned+ensembled in the simulator (e.g. only 1 config).
methods_no_hpo = [
    "TabDPT_GPU",
]

# Methods fit with bagging (8-fold); if absent, the model could instead have been refit on full data.
methods_is_bag = [
    "Dummy",
    "ExplainableBM",
    "LinearModel",
    "RealMLP_GPU",
]

methods_ag_key_map = {
    "Dummy": "DUMMY",
    "ExplainableBM": "EBM",
    "KNeighbors": "KNN",
    "LinearModel": "LR",
    "RealMLP_GPU": "REALMLP",
    "TabDPT_GPU": "TABDPT",
    "TabPFNv2_GPU": "TABPFNV2",
}

methods_display_name_map = {
    "Dummy": "Dummy",
    "ExplainableBM": "EBM",
    "KNeighbors": "KNN",
    "LinearModel": "Linear",
    "RealMLP_GPU": "RealMLP",
    "TabDPT_GPU": "TabDPT",
    "TabPFNv2_GPU": "TabPFNv2",
}

methods_config_default_map = {
    "Dummy": "Dummy_c1_BAG_L1",
    "ExplainableBM": "ExplainableBM_c1_BAG_L1",
    "KNeighbors": "KNeighbors_c1_BAG_L1",
    "LinearModel": "LinearModel_c1_BAG_L1",
    "RealMLP_GPU": "RealMLP_GPU_c1_BAG_L1",
    "TabDPT_GPU": "TabDPT_GPU_c1_BAG_L1",
    "TabPFNv2_GPU": "TabPFNv2_GPU_c1_BAG_L1",
}

methods_compute_map = {
    "Dummy": "cpu",
    "ExplainableBM": "cpu",
    "KNeighbors": "cpu",
    "LinearModel": "cpu",
    "RealMLP_GPU": "gpu",
    "TabDPT_GPU": "gpu",
    "TabPFNv2_GPU": "gpu",
}

methods_to_url_map = {
    "ExplainableBM": "https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf",
    "KNeighbors": "https://scikit-learn.org/stable/modules/neighbors.html",
    "LinearModel": "https://scikit-learn.org/stable/modules/linear_model.html",
    "RealMLP_GPU": "https://arxiv.org/abs/2407.04491",
    "TabDPT_GPU": "https://arxiv.org/abs/2410.18164",
    "TabPFNv2_GPU": "https://www.nature.com/articles/s41586-024-08328-6",
}

# Date each method was first introduced (paper or library release) — for the methods built by
# the loop below. `Dummy` is intentionally unset (no invention date).
methods_to_date_introduced_map = {
    "ExplainableBM": "2019-09",
    "KNeighbors": "1951",
    "LinearModel": "1958",
    "RealMLP_GPU": "2024-07",
    "TabDPT_GPU": "2024-10",
    "TabPFNv2_GPU": "2025-01",  # TabPFN v2 (Nature s41586-024-08328-6)
}

# Curated name lists (consumed by the 2025-06-12 collections in `methods.py`); these span the full
# 2025-06-12 method set, including the per-model-`info.py` models seeded above.
methods_main_paper = [
    "CatBoost",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "RealMLP",
    "XGBoost",
    "ModernNCA_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
    "AutoGluon_v130",
    # "Portfolio-N200-4h",
]

methods_gpu_ablation = [
    "ModernNCA",
    "TabM",
    "RealMLP_GPU",
]

for method in methods:
    compute_type = methods_compute_map[method]
    ag_key = methods_ag_key_map[method]
    config_default = methods_config_default_map[method]
    is_bag = method in methods_is_bag
    display_name = methods_display_name_map.get(method)
    assert compute_type in ["cpu", "gpu"]
    if compute_type == "cpu":
        method_kwargs = cpu_kwargs
    else:
        method_kwargs = gpu_kwargs
    if method in methods_no_hpo:
        can_hpo = False
    else:
        can_hpo = True

    reference_url = methods_to_url_map.get(method)

    method_kwargs = copy.deepcopy(method_kwargs)

    method_metadata = MethodMetadata.tabarena_legacy_s3(
        method=method,
        config_default=config_default,
        display_name=display_name,
        ag_key=ag_key,
        is_bag=is_bag,
        can_hpo=can_hpo,
        reference_url=reference_url,
        date_introduced=methods_to_date_introduced_map.get(method),
        **method_kwargs,
    )
    methods_2025_06_12.append(method_metadata)


# Non-model baselines / portfolio live in `tabarena.baselines.info`.
from tabarena.baselines.info import ag_130_metadata, portfolio_metadata

methods_2025_06_12.append(ag_130_metadata)
methods_2025_06_12.append(portfolio_metadata)
