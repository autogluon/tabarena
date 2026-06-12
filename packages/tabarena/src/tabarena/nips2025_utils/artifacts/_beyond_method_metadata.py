"""Method metadata for the Beyond-IID benchmark (``beyond_iid_benchmark_2026``).

The Beyond runs reuse tabarena's model wrappers and search spaces; only the
result-artifact metadata differs (artifact name, ``TA-*`` method names for the
re-run models, CPU/GPU compute, config defaults). Kept as standalone
``MethodMetadata`` instances in the dated-artifacts style — the per-model
``info.py`` modules carry the *tabarena* benchmark's metadata, not Beyond's.

``beyond_method_metadata_collection`` is the aggregate consumed by
:class:`~tabarena.nips2025_utils.beyond_arena_context.BeyondArenaContext`.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._method_metadata_collection import MethodMetadataCollection

_common_kwargs = dict(
    method_type="config",
    artifact_name="beyond_iid_benchmark_2026",
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)

_common_bag_kwargs = dict(
    can_hpo=True,
    is_bag=True,
    verified=True,
    **_common_kwargs,
)

_common_fm_kwargs = dict(
    compute="gpu",
    can_hpo=False,
    is_bag=False,
    **_common_kwargs,
)

# -- Classic (CPU, bagged, HPO-able) methods ----------------------------------------
beyond_linear_metadata = MethodMetadata(
    method="LinearModel",
    display_name="Linear",
    compute="cpu",
    ag_key="LR",
    config_default="LinearModel_c1_BAG_L1",
    reference_url="https://scikit-learn.org/stable/modules/linear_model.html",
    **_common_bag_kwargs,
)
beyond_random_forest_metadata = MethodMetadata(
    method="RandomForest",
    display_name="RandomForest",
    compute="cpu",
    ag_key="RF",
    config_default="RandomForest_c1_BAG_L1",
    reference_url="https://link.springer.com/article/10.1023/A:1010933404324",
    **_common_bag_kwargs,
)
beyond_extra_trees_metadata = MethodMetadata(
    method="ExtraTrees",
    display_name="ExtraTrees",
    compute="cpu",
    ag_key="XT",
    config_default="ExtraTrees_c1_BAG_L1",
    reference_url="https://link.springer.com/article/10.1007/s10994-006-6226-1",
    **_common_bag_kwargs,
)
beyond_catboost_metadata = MethodMetadata(
    method="CatBoost",
    display_name="CatBoost",
    compute="cpu",
    ag_key="CAT",
    config_default="CatBoost_c1_BAG_L1",
    reference_url="https://arxiv.org/abs/1706.09516",
    **_common_bag_kwargs,
)
beyond_lightgbm_metadata = MethodMetadata(
    method="LightGBM",
    display_name="LightGBM",
    compute="cpu",
    ag_key="GBM",
    config_default="LightGBM_c1_BAG_L1",
    reference_url="https://papers.nips.cc/paper_files/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html",
    **_common_bag_kwargs,
)
beyond_xgboost_metadata = MethodMetadata(
    method="XGBoost",
    display_name="XGBoost",
    compute="cpu",
    ag_key="XGB",
    config_default="XGBoost_c1_BAG_L1",
    reference_url="https://arxiv.org/abs/1603.02754",
    **_common_bag_kwargs,
)

# -- GPU neural / deep methods (bagged, HPO-able) ------------------------------------
beyond_realmlp_metadata = MethodMetadata(
    method="TA-RealMLP",
    display_name="RealMLP",
    compute="gpu",
    ag_key="TA-REALMLP",
    config_default="TA-RealMLP_c1_BAG_L1",
    reference_url="https://arxiv.org/abs/2407.04491",
    **_common_bag_kwargs,
)
beyond_tabm_metadata = MethodMetadata(
    method="TA-TabM",
    display_name="TabM",
    compute="gpu",
    ag_key="TA-TABM",
    config_default="TA-TabM_c1_BAG_L1",
    reference_url="https://arxiv.org/abs/2410.24210",
    **_common_bag_kwargs,
)

# -- Foundation models (GPU, no HPO, unbagged) ---------------------------------------
beyond_tabdpt_metadata = MethodMetadata(
    method="TA-TabDPT",
    display_name="TabDPT",
    ag_key="TA-TABDPT",
    config_default="TA-TabDPT_c1_BAG_L1",
    verified=True,
    reference_url="https://arxiv.org/abs/2410.18164",
    **_common_fm_kwargs,
)
beyond_tabpfnv26_metadata = MethodMetadata(
    method="TA-TabPFN-2.6",
    display_name="TabPFN-2.6",
    ag_key="TA-TABPFN-2.6",
    config_default="TA-TabPFN-2.6_c1_BAG_L1",
    verified=True,
    reference_url="https://arxiv.org/abs/2511.08667",
    **_common_fm_kwargs,
)
beyond_tabiclv2_metadata = MethodMetadata(
    method="TA-TabICLv2",
    display_name="TabICLv2",
    ag_key="TA-TABICLv2",
    config_default="TA-TabICLv2_c1_BAG_L1",
    verified=True,
    reference_url="https://arxiv.org/abs/2602.11139",
    **_common_fm_kwargs,
)
beyond_method_metadata_lst: list[MethodMetadata] = [
    beyond_linear_metadata,
    beyond_random_forest_metadata,
    beyond_extra_trees_metadata,
    beyond_catboost_metadata,
    beyond_lightgbm_metadata,
    beyond_xgboost_metadata,
    beyond_realmlp_metadata,
    beyond_tabm_metadata,
    beyond_tabdpt_metadata,
    beyond_tabpfnv26_metadata,
    beyond_tabiclv2_metadata,
]

beyond_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=beyond_method_metadata_lst,
)
