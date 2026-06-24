from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.iltm.hpo import gen_iltm
from tabarena.models.iltm.model import ILTMModel

iltm_method_metadata = MethodMetadata.config(
    method="iLTM",
    suite="tabarena-2026-05-13",
    ag_key="TA-ILTM",
    config_default="iLTM_c1_BAG_L1",
    compute="gpu",
    is_bag=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    date="2026-05-29",
    reference_url="https://arxiv.org/abs/2511.15941",
    display_name="iLTM",
)


iltm_info = ModelInfo(
    model_cls=ILTMModel,
    search_space=gen_iltm,
    method_metadata=iltm_method_metadata,
    pip_extra=("iltm>=0.1.1",),
)
