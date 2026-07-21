from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabdpt.hpo import gen_tabdpt, gen_tabdpt_turbo
from tabarena.models.tabdpt.model import TabDPTModel, TabDPTTurboModel

tabdpt_descriptor = ModelDescriptor(
    display_name="TabDPT",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2410.18164",
    date_introduced="2024-10",
)

tabdpt_turbo_descriptor = ModelDescriptor(
    display_name="TabDPT-Turbo",
    compute="gpu",
    is_bag=False,
    reference_url="https://openreview.net/pdf?id=Y00pwFyrHR",
)

tabdpt_method_metadata = tabdpt_descriptor.method_metadata(
    method="TabDPT_GPU",
    suite="tabarena-2025-10-20",
    ag_key="TABDPT",
    model_key="TABDPT_GPU",
    config_default="TabDPT_GPU_c1_BAG_L1",
    cache_type="s3",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache", "upload_as_public": True},
    date="2025-10-20",
)

# A distinct `model_key`/`ag_key` keeps TabDPT-Turbo (v1.2) a separate leaderboard method from TabDPT above.
tabdpt_turbo_method_metadata = tabdpt_turbo_descriptor.method_metadata(
    method="TabDPT-Turbo",
    suite="tabarena-2026-07-13",
    ag_key="TA-TABDPT-TURBO",
    model_key="TABDPT_TURBO",
    config_default="TabDPT-Turbo_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-07-15",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabdpt_info = ModelInfo(
    model_cls=TabDPTModel,
    search_space=gen_tabdpt,
    method_metadata=tabdpt_method_metadata,
    pip_extra=("tabdpt>=1.2.0",),
    prefetch_weights=TabDPTModel.prefetch_weights,
)


tabdpt_turbo_info = ModelInfo(
    model_cls=TabDPTTurboModel,
    search_space=gen_tabdpt_turbo,
    method_metadata=tabdpt_turbo_method_metadata,
    pip_extra=("tabdpt>=1.2.0",),
    prefetch_weights=TabDPTTurboModel.prefetch_weights,
)
