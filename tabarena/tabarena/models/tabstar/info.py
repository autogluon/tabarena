from __future__ import annotations

from tabarena.models.tabstar.model import TabSTARModel
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabstar.hpo import gen_tabstar
from tabarena.models._method_metadata import MethodMetadata

tabstar_method_metadata = MethodMetadata(
    method="TabSTAR",
    method_type="config",
    display_name="TabSTAR",
    compute="gpu",
    date="2026-03-02",
    ag_key="TABSTAR",
    model_key="TABSTAR",
    config_default="TabSTAR_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=True,
    has_processed=True,
    artifact_name="tabarena-2026-03-18",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_results=True,
    name_suffix=None,
    verified=True,
    reference_url="https://arxiv.org/abs/2505.18125",
    cache_type="r2",
)


tabstar_info = ModelInfo(
    model_cls=TabSTARModel,
    search_space=gen_tabstar,
    method_metadata=tabstar_method_metadata,
    pip_extra=("tabstar==1.1.15",),
)
