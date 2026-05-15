from __future__ import annotations

from tabarena.benchmark.models.ag.tabicl.tabicl_model import TabICLv2Model
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabicl.hpo import gen_tabiclv2
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata


tabiclv2_method_metadata = MethodMetadata(
    method="TabICLv2",
    method_type="config",
    display_name="TabICLv2",
    compute="gpu",
    date="2026-02-16",
    ag_key="TABICLV2",
    model_key="TABICLV2",
    config_default="TabICLv2_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-02-16",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2602.11139",
)


tabiclv2_info = ModelInfo(
    model_cls=TabICLv2Model,
    search_space=gen_tabiclv2,
    method_metadata=tabiclv2_method_metadata,
)


# Note: the older `TabICLModel` (gen_tabicl) shares this folder for search-space
# co-location but does not yet have a dedicated `MethodMetadata` entry —
# it's part of the `methods_2025_06_12` factory list, not the per-model registry.
