from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.contexts.tabarena._tabarena_method_metadata_2025_06_12 import (
    methods_2025_06_12,
    methods_gpu_ablation,
    methods_main_paper,
)
from tabarena.contexts.tabarena._tabarena_method_metadata_2025_09_03 import (
    ag_140_metadata,
    betatabpfn_metadata,
    tabflex_metadata,
)
from tabarena.contexts.tabarena._tabarena_method_metadata_2025_10_20 import (
    portfolio_metadata_paper_cr,
)
from tabarena.contexts.tabarena._tabarena_method_metadata_2025_11_01 import (
    ag_140_bq_4h8c_metadata,
    methods_2025_11_01_ag,
)
from tabarena.contexts.tabarena._tabarena_method_metadata_2025_12_18 import ag_150_eq_4h8c_metadata
from tabarena.contexts.tabarena._tabarena_method_metadata_2026_01_23_tabprep import (
    tabprep_gbm_metadata,
    tabprep_lr_metadata,
    tabprep_realtabpfnv250_metadata,
    tabprep_tabm_metadata,
)
from tabarena.contexts.tabarena._tabarena_method_metadata_misc import (
    gbm_aio_0808_metadata,
    # prep_gbm_v6_metadata,
)
from tabarena.models._method_metadata_collection import MethodMetadataCollection

# Per-model `info.py` is the canonical source for each method's `MethodMetadata`; the collection
# below references these directly. (Some keep their historical alias, used by the per-suite full
# lists / downstream collections.)
from tabarena.models.catboost.info import catboost_new_method_metadata
from tabarena.models.chimeraboost.info import (
    chimeraboost_method_metadata,
    chimeraboost_new_method_metadata,
)
from tabarena.models.ebm.info import (
    ebm_method_metadata as ebm_metadata,
    ebm_new_method_metadata,
)
from tabarena.models.extra_trees.info import extra_trees_new_method_metadata
from tabarena.models.fastai.info import fastai_method_metadata
from tabarena.models.iltm.info import iltm_method_metadata
from tabarena.models.knn.info import knn_method_metadata as knn_metadata
from tabarena.models.lightgbm.info import lightgbm_method_metadata
from tabarena.models.limix.info import limix_method_metadata as limix_metadata
from tabarena.models.lr.info import lr_method_metadata as lr_metadata
from tabarena.models.mitra.info import mitra_method_metadata as mitra_metadata
from tabarena.models.modernnca.info import modernnca_gpu_method_metadata
from tabarena.models.nn_torch.info import nn_torch_method_metadata
from tabarena.models.nori.info import nori30m_method_metadata, nori_method_metadata, nori_new_method_metadata
from tabarena.models.orionmsp.info import orionmsp_method_metadata as orionmsp_metadata
from tabarena.models.perpetual_booster.info import (
    perpetual_booster_method_metadata as perpetualbooster_metadata,
)
from tabarena.models.random_forest.info import random_forest_method_metadata
from tabarena.models.realmlp.info import realmlp_method_metadata as realmlp_gpu_metadata
from tabarena.models.sap_rpt_oss.info import (
    sap_rpt_oss_method_metadata as contexttab_metadata,
)
from tabarena.models.tabdpt.info import (
    tabdpt_method_metadata as tabdpt_metadata,
    tabdpt_turbo_method_metadata,
)
from tabarena.models.tabfm.info import tabfm_method_metadata, tabfm_new_method_metadata
from tabarena.models.tabicl.info import (
    tabicl_method_metadata,
    tabiclv2_method_metadata as tabiclv2_metadata,
    tabiclv2_new_method_metadata,
)
from tabarena.models.tabm.info import tabm_new_method_metadata
from tabarena.models.tabpfn_3.info import (
    tabpfn_3_method_metadata as tabpfnv3_method_metadata,
    tabpfn_3_new_method_metadata,
)
from tabarena.models.tabpfnv2_5.info import (
    realtabpfnv25_method_metadata as realtabpfn25_metadata,
    tabpfnv26_method_metadata as tabpfn26_metadata,
)
from tabarena.models.tabstar.info import tabstar_method_metadata as tabstar_metadata
from tabarena.models.tabswift.info import tabswift_method_metadata, tabswift_new_method_metadata
from tabarena.models.xgboost.info import xgboost_method_metadata
from tabarena.models.xrfm.info import xrfm_method_metadata as xrfm_metadata

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

# TabPFNv2_GPU has no per-model info.py (it is built by the 2025-06-12 factory); pull it out by name.
tabpfnv2_gpu_metadata = next(m for m in methods_2025_06_12 if m.method == "TabPFNv2_GPU")

# Per-suite full lists — feed the complete (historical) collection below.
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
    # prep_gbm_v6_metadata,
]

methods_tabprep = [
    tabprep_gbm_metadata,
    tabprep_lr_metadata,
    tabprep_realtabpfnv250_metadata,  # only first 3 splits
    tabprep_tabm_metadata,  # only first 3 splits
]

updated_methods_camera_ready = [
    "LinearModel",
    "KNeighbors",
    "Portfolio-N200-4h",
]
methods_2025_10_20_camera_ready = [
    m for m in methods_2025_06_12 if m.method not in updated_methods_camera_ready
] + methods_2025_10_20


# The latest results for each method — exactly the methods in the TabArena paper (one per method),
# each referencing its per-model `info.py` MethodMetadata directly. TabArenaContext uses this
# collection as-is (no separate name allowlist). To add a newly processed/uploaded method, import its
# `info.py` metadata above and add it here under the appropriate group.
tabarena_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=[
        # AutoGluon
        ag_140_bq_4h8c_metadata,
        ag_150_eq_4h8c_metadata,
        # Default tabular models (CPU)
        catboost_new_method_metadata,
        chimeraboost_new_method_metadata,
        ebm_new_method_metadata,
        extra_trees_new_method_metadata,
        knn_metadata,
        lightgbm_method_metadata,
        lr_metadata,
        fastai_method_metadata,
        nn_torch_method_metadata,
        random_forest_method_metadata,
        xgboost_method_metadata,
        # Neural / GPU / foundation models
        mitra_metadata,
        modernnca_gpu_method_metadata,
        realmlp_gpu_metadata,
        tabdpt_metadata,
        tabdpt_turbo_method_metadata,
        tabicl_method_metadata,
        tabm_new_method_metadata,
        tabpfnv2_gpu_metadata,
        xrfm_metadata,
        betatabpfn_metadata,
        tabflex_metadata,
        realtabpfn25_metadata,
        contexttab_metadata,
        tabiclv2_new_method_metadata,
        tabstar_metadata,
        perpetualbooster_metadata,
        tabpfn26_metadata,
        limix_metadata,
        orionmsp_metadata,
        tabpfn_3_new_method_metadata,
        iltm_method_metadata,
        nori_new_method_metadata,
        tabfm_new_method_metadata,
        tabswift_new_method_metadata,
    ],
)

# Superseded predecessors of the tabarena-2026-07-13 reruns that appear in no dated suite list above.
methods_superseded_2026_07_13: list[MethodMetadata] = [
    chimeraboost_method_metadata,
    nori_method_metadata,
    nori30m_method_metadata,
        tabfm_method_metadata,
        tabswift_method_metadata,
    tabpfnv3_method_metadata,
    tabiclv2_metadata,
]

# All historical results for each method = the latest collection + every superseded variant: the
# replaced / CPU-only / extra-AutoGluon-preset / non-paper entries from the dated suite lists (and
# misc) that the collection above does not keep.
_collection_methods = tabarena_method_metadata_collection.method_metadata_lst
methods_historical = [
    m
    for m in [
        *methods_2025_06_12,
        *methods_2025_09_03,
        *methods_2025_10_20,
        *methods_2025_11_01_ag,
        *methods_misc,
        *methods_superseded_2026_07_13,
    ]
    if m not in _collection_methods
]
tabarena_method_metadata_complete_collection = tabarena_method_metadata_collection.with_additional_methods(
    methods_historical,
)

# All historical results for each method
tabarena_method_metadata_2025_06_12_collection = MethodMetadataCollection(
    method_metadata_lst=methods_2025_10_20_camera_ready,
)

tabarena_method_metadata_2025_06_12_collection_main = MethodMetadataCollection(
    method_metadata_lst=[
        m for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst if m.method in methods_main_paper
    ]
)

tabarena_method_metadata_2025_06_12_collection_gpu_ablation = MethodMetadataCollection(
    method_metadata_lst=[
        m
        for m in tabarena_method_metadata_2025_06_12_collection.method_metadata_lst
        if m.method in methods_gpu_ablation
    ]
)
