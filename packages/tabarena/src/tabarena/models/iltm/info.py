from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.iltm.hpo import gen_iltm
from tabarena.models.iltm.model import ILTMModel

iltm_method_metadata = MethodMetadata(
    method="iLTM",
    artifact_name="tabarena-2026-05-13",
    method_type="config",
    display_name="iLTM",
    compute="gpu",
    ag_key="TA-ILTM",
    config_default="iLTM_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    verified=True,
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    cache_type="r2",
    reference_url="https://arxiv.org/abs/2511.15941",
)


iltm_info = ModelInfo(
    model_cls=ILTMModel,
    search_space=gen_iltm,
    method_metadata=iltm_method_metadata,
    pip_extra=("iltm>=0.1.1",),
)
