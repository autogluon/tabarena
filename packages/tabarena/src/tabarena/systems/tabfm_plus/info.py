from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models.tabfm.model import prefetch_weights
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.tabfm_plus.hpo import gen_tabfm_plus
from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel

tabfm_plus_method_metadata = MethodMetadata.system(
    method="TabFM+",
    name="TabFM+",
    suite="tabarena-2026-06-26",
    compute="gpu",
    date="2026-06-26",
    date_introduced="2026-06-30",
    reference_url="https://github.com/google-research/tabfm",
    verified=False,
)


tabfm_plus_info = SystemInfo(
    system_cls=TabFMPlusSystemModel,
    config_generator=gen_tabfm_plus,
    method_metadata=tabfm_plus_method_metadata,
    # Shares TabFM's checkpoint and its install, so the extra matches the model's.
    pip_extra=(
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@633cd265f498e1d20c9625be0639f6305d8e2541",
    ),
    prefetch_weights=prefetch_weights,
)
