"""Method metadata for the Beyond-IID benchmark (``beyond_iid_benchmark_2026``).

The Beyond runs reuse tabarena's model wrappers and search spaces; only the
result-artifact metadata differs (artifact name, ``TA-*`` method names and
``ag_key`` re-keying for the re-run models, per-run ``config_default`` and
``can_hpo``). The *intrinsic* model facts (display name, paper URL, compute
class, whether it bags) are not re-typed here: each entry is built from the
model's shared :class:`~tabarena.models._method_metadata.ModelDescriptor`,
declared once in that model's ``models/<key>/info.py`` and reused for both the
tabarena run and this Beyond re-run.

``beyond_method_metadata_collection`` is the aggregate consumed by
:class:`~tabarena.evaluation.context.beyond_arena.BeyondArenaContext`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.models._method_metadata_collection import MethodMetadataCollection
from tabarena.models.catboost.info import catboost_descriptor
from tabarena.models.extra_trees.info import extra_trees_descriptor
from tabarena.models.lightgbm.info import lightgbm_descriptor
from tabarena.models.lr.info import lr_descriptor
from tabarena.models.random_forest.info import random_forest_descriptor
from tabarena.models.realmlp.info import realmlp_descriptor
from tabarena.models.tabdpt.info import tabdpt_descriptor
from tabarena.models.tabicl.info import tabiclv2_descriptor
from tabarena.models.tabm.info import tabm_descriptor
from tabarena.models.tabpfn_3.info import tabpfn_3_descriptor
from tabarena.models.tabpfnv2_5.info import tabpfnv26_descriptor
from tabarena.models.xgboost.info import xgboost_descriptor

if TYPE_CHECKING:
    from tabarena.models._method_metadata import MethodMetadata

_common_kwargs = dict(
    method_type="config",
    artifact_name="beyond_iid_benchmark_2026",
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
    cache_type="r2",
    s3_bucket="tabarena",
    s3_prefix="cache",
)

# Bagged (HPO-able) and foundation-model presets. ``compute`` / ``is_bag`` are *not* set here:
# they are intrinsic to the model and come from each entry's ModelDescriptor.
_common_bag_kwargs = dict(
    can_hpo=True,
    verified=True,
    **_common_kwargs,
)

_common_fm_kwargs = dict(
    can_hpo=False,
    **_common_kwargs,
)

# -- Classic (CPU, bagged, HPO-able) methods ----------------------------------------
beyond_linear_metadata = lr_descriptor.method_metadata(
    method="LinearModel",
    ag_key="LR",
    config_default="LinearModel_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_random_forest_metadata = random_forest_descriptor.method_metadata(
    method="RandomForest",
    ag_key="RF",
    config_default="RandomForest_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_extra_trees_metadata = extra_trees_descriptor.method_metadata(
    method="ExtraTrees",
    ag_key="XT",
    config_default="ExtraTrees_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_catboost_metadata = catboost_descriptor.method_metadata(
    method="CatBoost",
    ag_key="CAT",
    config_default="CatBoost_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_lightgbm_metadata = lightgbm_descriptor.method_metadata(
    method="LightGBM",
    ag_key="GBM",
    config_default="LightGBM_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_xgboost_metadata = xgboost_descriptor.method_metadata(
    method="XGBoost",
    ag_key="XGB",
    config_default="XGBoost_c1_BAG_L1",
    **_common_bag_kwargs,
)

# -- GPU neural / deep methods (bagged, HPO-able) ------------------------------------
beyond_realmlp_metadata = realmlp_descriptor.method_metadata(
    method="TA-RealMLP",
    ag_key="TA-REALMLP",
    config_default="TA-RealMLP_c1_BAG_L1",
    **_common_bag_kwargs,
)
beyond_tabm_metadata = tabm_descriptor.method_metadata(
    method="TA-TabM",
    ag_key="TA-TABM",
    config_default="TA-TabM_c1_BAG_L1",
    **_common_bag_kwargs,
)

# -- Foundation models (GPU, no HPO, unbagged) ---------------------------------------
beyond_tabdpt_metadata = tabdpt_descriptor.method_metadata(
    method="TA-TabDPT",
    ag_key="TA-TABDPT",
    config_default="TA-TabDPT_c1_BAG_L1",
    verified=True,
    **_common_fm_kwargs,
)
beyond_tabpfnv26_metadata = tabpfnv26_descriptor.method_metadata(
    method="TA-TabPFN-2.6",
    ag_key="TA-TABPFN-2.6",
    config_default="TA-TabPFN-2.6_c1_BAG_L1",
    verified=True,
    **_common_fm_kwargs,
)
beyond_tabiclv2_metadata = tabiclv2_descriptor.method_metadata(
    method="TA-TabICLv2",
    ag_key="TA-TABICLv2",
    config_default="TA-TabICLv2_c1_BAG_L1",
    verified=True,
    **_common_fm_kwargs,
)
beyond_tabpfn3_metadata = tabpfn_3_descriptor.method_metadata(
    method="TA-TabPFN-3",
    ag_key="TA-TABPFN-3",
    config_default="TA-TabPFN-3_c1_BAG_L1",
    verified=True,
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
    beyond_tabpfn3_metadata,
]

beyond_method_metadata_collection = MethodMetadataCollection(
    method_metadata_lst=beyond_method_metadata_lst,
)
