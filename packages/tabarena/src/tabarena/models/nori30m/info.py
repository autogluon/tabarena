from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nori30m.hpo import gen_nori30m
from tabarena.models.nori30m.model import Nori30MModel

nori30m_method_metadata = MethodMetadata.config(
    method="Nori-30M",
    ag_key="TA-NORI-30M",
    config_default="Nori-30M_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-13",
    reference_url="https://huggingface.co/Synthefy/Nori-30M",
    display_name="Nori-30M",
    # verified=False and no cache: the result artifacts are hosted in the official pool by the
    # maintainers after a verified re-run (see the base Nori info once its results are hosted).
    verified=False,
)

nori30m_info = ModelInfo(
    model_cls=Nori30MModel,
    search_space=gen_nori30m,
    method_metadata=nori30m_method_metadata,
    pip_extra=("synthefy-nori>=0.10.0",),  # >=0.10.0 provides the model= variant selector
    prefetch_weights=Nori30MModel.prefetch_weights,
)
