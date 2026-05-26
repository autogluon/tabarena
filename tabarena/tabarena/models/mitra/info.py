from __future__ import annotations

from autogluon.tabular.models import MitraModel

from tabarena.models._model_info import ModelInfo
from tabarena.models.mitra.hpo import gen_mitra
from tabarena.models._method_metadata import MethodMetadata


mitra_method_metadata = MethodMetadata(
    method="Mitra_GPU",
    method_type="config",
    display_name="Mitra",
    compute="gpu",
    date="2025-09-03",
    ag_key="MITRA",
    model_key="MITRA_GPU",
    config_default="Mitra_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    verified=True,
    reference_url="https://arxiv.org/abs/2510.21204",
    artifact_name="tabarena-2025-09-03",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    name_suffix=None,
)


mitra_info = ModelInfo(
    model_cls=MitraModel,
    search_space=gen_mitra,
    method_metadata=mitra_method_metadata,
)
