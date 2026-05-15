from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabarena.nips2025_utils.artifacts.method_metadata_collection import MethodMetadataCollection

# Per-model `info.py` is the canonical source for each migrated model's
# `MethodMetadata`. Aliased here under the historical name so the rest of
# the aggregator (and downstream collections) keeps working unchanged.
from tabarena.models.ebm.info import ebm_method_metadata as ebm_metadata
from tabarena.models.knn.info import knn_method_metadata as knn_metadata
from tabarena.models.limix.info import limix_method_metadata as limix_metadata
from tabarena.models.lr.info import lr_method_metadata as lr_metadata
from tabarena.models.mitra.info import mitra_method_metadata as mitra_metadata
from tabarena.models.orionmsp.info import orionmsp_method_metadata as orionmsp_metadata
from tabarena.models.perpetual_booster.info import (
    perpetual_booster_method_metadata as perpetualbooster_metadata,
)
from tabarena.models.realmlp.info import realmlp_method_metadata as realmlp_gpu_metadata
from tabarena.models.sap_rpt_oss.info import (
    sap_rpt_oss_method_metadata as contexttab_metadata,
)
from tabarena.models.tabdpt.info import tabdpt_method_metadata as tabdpt_metadata
from tabarena.models.tabicl.info import (
    tabiclv2_method_metadata as tabiclv2_metadata,
)
from tabarena.models.tabpfn_3.info import (
    tabpfn_3_method_metadata as tabpfnv3_method_metadata,
)
from tabarena.models.tabpfnv2_5.info import (
    realtabpfnv25_method_metadata as realtabpfn25_metadata,
    tabpfnv26_method_metadata as tabpfn26_metadata,
)
from tabarena.models.tabstar.info import tabstar_method_metadata as tabstar_metadata
from tabarena.models.xrfm.info import xrfm_method_metadata as xrfm_metadata

from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_06_12 import (
    methods_2025_06_12,
    methods_main_paper,
    methods_gpu_ablation,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_09_03 import (
    ag_140_metadata,
    tabflex_metadata,
    betatabpfn_metadata,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_10_20 import (
    portfolio_metadata_paper_cr,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_11_01 import (
    ag_140_bq_4h8c_metadata,
    ag_140_eq_4h8c_metadata,
    methods_2025_11_01_ag,
)
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2025_12_18 import ag_150_eq_4h8c_metadata
from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_misc import (
    gbm_aio_0808_metadata,
    # prep_gbm_v6_metadata,
)

from tabarena.nips2025_utils.artifacts._tabarena_method_metadata_2026_01_23_tabprep import (
    tabprep_gbm_metadata,
    tabprep_lr_metadata,
    tabprep_realtabpfnv250_metadata,
    tabprep_tabm_metadata,
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

methods_2025_09_03_keep: list[MethodMetadata] = [
    ebm_metadata,
    limix_metadata,
    mitra_metadata,
    realmlp_gpu_metadata,
    xrfm_metadata,
    betatabpfn_metadata,
    tabflex_metadata,
]

methods_2025_11_01_keep: list[MethodMetadata] = [
    ag_140_bq_4h8c_metadata,
    ag_140_eq_4h8c_metadata,
]

methods_2025_10_20: list[MethodMetadata] = [
    lr_metadata,
    knn_metadata,
    portfolio_metadata_paper_cr,
]

methods_2025_12_18: list[MethodMetadata] = [
    ag_150_eq_4h8c_metadata,
]

methods_misc: list[MethodMetadata] = [
    gbm_aio_0808_metadata,
    # prep_gbm_v6_metadata,
]

methods_tabprep = [
    tabprep_gbm_metadata,
    tabprep_lr_metadata,
    tabprep_realtabpfnv250_metadata,  # only first 3 splits
    tabprep_tabm_metadata,  # only first 3 splits
]

replaced_methods = [
    "ExplainableBM",
    "RealMLP_GPU",
    "TabDPT_GPU",
    "AutoGluon_v130",
]
updated_methods_camera_ready = [
    "LinearModel",
    "KNeighbors",
    "Portfolio-N200-4h",
]
replaced_methods += updated_methods_camera_ready
removed_cpu_methods = [
    "ModernNCA",
    "TabM",
    "RealMLP",
]

methods_2025_06_12_keep = [m for m in methods_2025_06_12 if m.method not in replaced_methods and m.method not in removed_cpu_methods]
methods_2025_10_20_camera_ready = [m for m in methods_2025_06_12 if m.method not in updated_methods_camera_ready] + methods_2025_10_20


# The latest results for each method
tabarena_method_metadata_collection = MethodMetadataCollection(method_metadata_lst=
    methods_2025_06_12_keep +
    methods_2025_09_03_keep +
    methods_2025_10_20 +
    methods_2025_11_01_keep +
    methods_2025_12_18 +
    [tabdpt_metadata] +
    [realtabpfn25_metadata] +
    [contexttab_metadata] +
    [tabiclv2_metadata] +
    [tabstar_metadata] +
    [perpetualbooster_metadata] +
    [tabpfn26_metadata] +
    [tabpfnv3_method_metadata] +
    [orionmsp_metadata] +
    methods_misc,
)

# All historical results for each method
tabarena_method_metadata_complete_collection = MethodMetadataCollection(method_metadata_lst=
    methods_2025_06_12 +
    methods_2025_09_03 +
    methods_2025_10_20 +
    methods_2025_11_01_ag +
    methods_2025_12_18 +
    [tabdpt_metadata] +
    [realtabpfn25_metadata] +
    [contexttab_metadata] +
    [tabiclv2_metadata] +
    [tabstar_metadata] +
    [perpetualbooster_metadata] +
    [tabpfn26_metadata] +
    [tabpfnv3_method_metadata] +
    [orionmsp_metadata] +
    methods_misc,
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
