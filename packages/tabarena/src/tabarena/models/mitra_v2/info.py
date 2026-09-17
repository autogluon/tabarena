from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.mitra_v2.hpo import gen_mitra_v2
from tabarena.models.mitra_v2.model import MitraV2Model, prefetch_weights

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
mitra_v2_method_metadata = MethodMetadata.config(
    method="Mitra-v2",
    suite="tabarena-2026-09-12",
    ag_key="TA-MITRA-V2",
    model_key="MITRA_V2",
    config_default="Mitra-v2_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=True,
    date="2026-09-12",
    date_introduced="2026-09-03",  # technical report date
    reference_url="https://arxiv.org/abs/2609.04540",
    license="Apache-2.0",
    display_name="Mitra-v2",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Rerun on the timing and warm-up pipeline (PR #584).
mitra_v2_new_method_metadata = MethodMetadata.config(
    method="Mitra-v2",
    validation_protocol="8x1",
    suite="tabarena-2026-09-16",
    ag_key="TA-MITRA-V2",
    model_key="MITRA_V2",
    config_default="Mitra-v2_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=True,
    date="2026-09-16",
    date_introduced="2026-09-03",  # technical report date
    reference_url="https://arxiv.org/abs/2609.04540",
    display_name="Mitra-v2",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    license="Apache-2.0",
)


mitra_v2_info = ModelInfo(
    model_cls=MitraV2Model,
    search_space=gen_mitra_v2,
    method_metadata=mitra_v2_new_method_metadata,
    # AutoGluon's Mitra extra (loguru, einops, transformers, huggingface_hub, ...). flash-attn is
    # optional and needs a prebuilt wheel: `pip install flash-attn --no-build-isolation`.
    pip_extra=("autogluon.tabular[mitra]>=1.6,<1.7",),
    prefetch_weights=prefetch_weights,
)
