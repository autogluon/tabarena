from __future__ import annotations

import copy

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

methods_2025_06_12 = []

common_kwargs = dict(
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="config",
    has_raw=True,
    has_processed=True,
    has_results=True,
    # FIXME: technically GBDTs and baselines are not verified
    verified=True,
)

s3_cache_kwargs = dict(
    s3_bucket="tabarena",
    s3_prefix="cache",
)

cpu_kwargs = dict(
    compute="cpu",
    name_suffix=None,
    **common_kwargs,
)

gpu_kwargs = dict(
    compute="gpu",
    name_suffix="_GPU",
    **common_kwargs,
)

# Methods in this list will upload s3 artifacts privately (useful for storing results for unreleased models)
# If a method is not in this list, it will be public when uploaded.
methods_upload_as_private = []


# If the method should not be tuned/tuned+enesmbled in the simulator, for example, due to having only 1 config
methods_no_hpo = [
    "TabICL_GPU",
    "TabDPT_GPU",
]

# If the method was fit with bagging (8-fold)
# If not present in this list, the model could instead have been refit on the full data, ex: TabPFNv2
methods_is_bag = [
    "CatBoost",
    "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    # "KNeighbors",
    "LightGBM",
    "LinearModel",
    "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "RealMLP",
    "TabM",
    "XGBoost",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    # "TabDPT_GPU",
    # "TabICL_GPU",
    "TabM_GPU",
    # "TabPFNv2_GPU",
]


methods_ag_key_map = {
    "CatBoost": "CAT",
    "Dummy": "DUMMY",
    "ExplainableBM": "EBM",
    "ExtraTrees": "XT",
    "KNeighbors": "KNN",
    "LightGBM": "GBM",
    "LinearModel": "LR",
    "ModernNCA": "MNCA",
    "NeuralNetFastAI": "FASTAI",
    "NeuralNetTorch": "NN_TORCH",
    "RandomForest": "RF",
    "RealMLP": "REALMLP",
    "TabM": "TABM",
    "XGBoost": "XGB",

    "ModernNCA_GPU": "MNCA",
    "RealMLP_GPU": "REALMLP",
    "TabDPT_GPU": "TABDPT",
    "TabICL_GPU": "TABICL",
    "TabM_GPU": "TABM",
    "TabPFNv2_GPU": "TABPFNV2",
}

methods_config_default_map = {
    "CatBoost": "CatBoost_c1_BAG_L1",
    "Dummy": "Dummy_c1_BAG_L1",
    "ExplainableBM": "ExplainableBM_c1_BAG_L1",
    "ExtraTrees": "ExtraTrees_c1_BAG_L1",
    "KNeighbors": "KNeighbors_c1_BAG_L1",
    "LightGBM": "LightGBM_c1_BAG_L1",
    "LinearModel": "LinearModel_c1_BAG_L1",
    "ModernNCA": "ModernNCA_c1_BAG_L1",
    "NeuralNetFastAI": "NeuralNetFastAI_c1_BAG_L1",
    "NeuralNetTorch": "NeuralNetTorch_c1_BAG_L1",
    "RandomForest": "RandomForest_c1_BAG_L1",
    "RealMLP": "RealMLP_c1_BAG_L1",
    "TabM": "TabM_c1_BAG_L1",
    "XGBoost": "XGBoost_c1_BAG_L1",

    "ModernNCA_GPU": "ModernNCA_GPU_c1_BAG_L1",
    "RealMLP_GPU": "RealMLP_GPU_c1_BAG_L1",
    "TabDPT_GPU": "TabDPT_GPU_c1_BAG_L1",
    "TabICL_GPU": "TabICL_GPU_c1_BAG_L1",
    "TabM_GPU": "TabM_GPU_c1_BAG_L1",
    "TabPFNv2_GPU": "TabPFNv2_GPU_c1_BAG_L1",
}


methods_compute_map = {
    "CatBoost": "cpu",
    "Dummy": "cpu",
    "ExplainableBM": "cpu",
    "ExtraTrees": "cpu",
    "KNeighbors": "cpu",
    "LightGBM": "cpu",
    "LinearModel": "cpu",
    "ModernNCA": "cpu",
    "NeuralNetFastAI": "cpu",
    "NeuralNetTorch": "cpu",
    "RandomForest": "cpu",
    "RealMLP": "cpu",
    "TabM": "cpu",
    "XGBoost": "cpu",

    "ModernNCA_GPU": "gpu",
    "RealMLP_GPU": "gpu",
    "TabDPT_GPU": "gpu",
    "TabICL_GPU": "gpu",
    "TabM_GPU": "gpu",
    "TabPFNv2_GPU": "gpu",

}

methods = [
    "CatBoost",
    "Dummy",
    "ExplainableBM",
    "ExtraTrees",
    "KNeighbors",
    "LightGBM",
    "LinearModel",
    "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RandomForest",
    "RealMLP",
    "TabM",
    "XGBoost",

    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
]

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
    "Portfolio-N200-4h",
]

methods_gpu_ablation = [
    "ModernNCA",
    "TabM",
    "RealMLP_GPU",
]

verified_methods = [
    "ExplainableBM",
    "ModernNCA",
    "NeuralNetFastAI",
    "NeuralNetTorch",
    "RealMLP",
    "TabM",
    "ModernNCA_GPU",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "TabICL_GPU",
    "TabM_GPU",
    "TabPFNv2_GPU",
]

methods_to_url_map = {
    "RealMLP_GPU": "https://arxiv.org/abs/2407.04491",
    "RealMLP": "https://arxiv.org/abs/2407.04491",
    "TabDPT_GPU": "https://arxiv.org/abs/2410.18164",
    "TabM_GPU": "https://arxiv.org/abs/2410.24210",
    "TabM": "https://arxiv.org/abs/2410.24210",
    "LightGBM": "https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    "CatBoost": "https://arxiv.org/abs/1706.09516",
    "ModernNCA_GPU": "https://arxiv.org/abs/2407.03257",
    "ModernNCA": "https://arxiv.org/abs/2407.03257",
    "XGBoost": "https://arxiv.org/abs/1603.02754",
    "TabPFNv2_GPU": "https://www.nature.com/articles/s41586-024-08328-6",
    "TabICL_GPU": "https://arxiv.org/abs/2502.05564",
    "NeuralNetTorch": "https://arxiv.org/abs/2003.06505",
    "NeuralNetFastAI": "https://arxiv.org/abs/2003.06505",
    "ExplainableBM": "https://www.cs.cornell.edu/~yinlou/papers/lou-kdd13.pdf",
    "RandomForest": "https://link.springer.com/article/10.1023/A:1010933404324",
    "KNeighbors": "https://scikit-learn.org/stable/modules/neighbors.html",
    "ExtraTrees": "https://link.springer.com/article/10.1007/s10994-006-6226-1",
    "LinearModel": "https://scikit-learn.org/stable/modules/linear_model.html",
}

tabarena_method_metadata_map_2025_06_12: dict[str, MethodMetadata] = {}

for method in methods:
    compute_type = methods_compute_map[method]
    ag_key = methods_ag_key_map[method]
    config_default = methods_config_default_map[method]
    is_bag = method in methods_is_bag
    upload_as_public = method not in methods_upload_as_private
    assert compute_type in ["cpu", "gpu"]
    if compute_type == "cpu":
        method_kwargs = cpu_kwargs
    else:
        method_kwargs = gpu_kwargs
    if method in methods_no_hpo:
        can_hpo = False
    else:
        can_hpo = True

    reference_url = methods_to_url_map.get(method, None)

    method_kwargs = copy.deepcopy(method_kwargs)

    method_metadata = MethodMetadata(
        method=method,
        config_default=config_default,
        ag_key=ag_key,
        is_bag=is_bag,
        can_hpo=can_hpo,
        upload_as_public=upload_as_public,
        reference_url=reference_url,
        **method_kwargs,
        **s3_cache_kwargs,
    )
    methods_2025_06_12.append(method_metadata)


ag_130_metadata = MethodMetadata(
    method="AutoGluon_v130",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="baseline",
    compute="cpu",
    has_raw=True,
    has_processed=True,
    has_results=True,
    upload_as_public=True,
    **s3_cache_kwargs,
)

methods_2025_06_12.append(ag_130_metadata)

portfolio_metadata = MethodMetadata(
    method="Portfolio-N200-4h",
    artifact_name="tabarena-2025-06-12",
    date="2025-06-12",
    method_type="portfolio",
    has_raw=False,
    has_processed=False,
    has_results=True,
    upload_as_public=True,
    **s3_cache_kwargs,
)

methods_2025_06_12.append(portfolio_metadata)
