from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.iltm.hpo import gen_iltm
from tabarena.models.iltm.model import ILTMModel

iltm_method_metadata = MethodMetadata.config(
    method="iLTM",
    date="2026-05-29",
    artifact_name="tabarena-2026-05-13",
    display_name="iLTM",
    compute="gpu",
    ag_key="TA-ILTM",
    config_default="iLTM_c1_BAG_L1",
    is_bag=True,
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    cache_type="r2",
    reference_url="https://arxiv.org/abs/2511.15941",
)


iltm_info = ModelInfo(
    model_cls=ILTMModel,
    search_space=gen_iltm,
    method_metadata=iltm_method_metadata,
    pip_extra=("iltm>=0.1.1",),
)
