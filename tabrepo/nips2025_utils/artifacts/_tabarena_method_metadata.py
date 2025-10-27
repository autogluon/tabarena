from __future__ import annotations

from tabrepo.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabrepo.nips2025_utils.artifacts.method_metadata_collection import MethodMetadataCollection
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_06_12 import (
    methods_2025_06_12,
    methods_main_paper,
    methods_gpu_ablation,
)

from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_09_03 import (
    ag_140_metadata,
    ebm_metadata,
    limix_metadata,
    mitra_metadata,
    realmlp_gpu_metadata,
    xrfm_metadata,
    tabflex_metadata,
    betatabpfn_metadata,
)
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_2025_10_20 import (
    lr_metadata,
    knn_metadata,
    portfolio_metadata_paper_cr
)
from tabrepo.nips2025_utils.artifacts._tabarena_method_metadata_misc import (
    gbm_aio_0808_metadata
)

methods_2025_09_03: list[MethodMetadata] = [
    ag_140_metadata,
    ebm_metadata,
    limix_metadata,
    mitra_metadata,
    realmlp_gpu_metadata,
    xrfm_metadata,
    betatabpfn_metadata,
    tabflex_metadata,
]

methods_2025_10_20: list[MethodMetadata] = [
    lr_metadata,
    knn_metadata,
    portfolio_metadata_paper_cr,
]

methods_misc: list[MethodMetadata] = [
    gbm_aio_0808_metadata,
]

replaced_methods = [
    "ExplainableBM",
    "RealMLP_GPU",
]
updated_methods_camera_ready = [
    "LinearModel",
    "KNeighbors",
    "Portfolio-N200-4h",
]
replaced_methods += updated_methods_camera_ready
methods_2025_06_12_keep = [m for m in methods_2025_06_12 if m.method not in replaced_methods]
methods_2025_10_20_camera_ready = [m for m in methods_2025_06_12 if m.method not in updated_methods_camera_ready] + methods_2025_10_20


# The latest results for each method
tabarena_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12_keep +
                        methods_2025_09_03 +
                        methods_2025_10_20 +
                        methods_misc,
)

# All historical results for each method
tabarena_method_metadata_complete_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_06_12 + methods_2025_09_03 + methods_2025_10_20 + methods_misc,
)

# All historical results for each method
tabarena_method_metadata_2025_06_12_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_10_20_camera_ready,
)

tabarena_method_metadata_2025_06_12_collection_main = MethodMetadataCollection(
    method_metadata_lst=[m for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst if m.method in methods_main_paper]
)

tabarena_method_metadata_2025_06_12_collection_gpu_ablation = MethodMetadataCollection(
    method_metadata_lst=[m for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst if m.method in methods_gpu_ablation]
)
