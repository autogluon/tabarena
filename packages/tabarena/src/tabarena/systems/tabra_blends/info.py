from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.tabra_blends.hpo import gen_tabra_blends
from tabarena.systems.tabra_blends.system import TabraBlendsSystemModel

tabra_blends_info = SystemInfo(
    system_cls=TabraBlendsSystemModel,
    config_generator=gen_tabra_blends,
    method_metadata=MethodMetadata.system(
        method="Tabra Blends",
        suite="tabarena-2026-11-21",
        compute="gpu",
        date="2026-11-21",
        date_introduced="2026-11",
        reference_url="https://github.com/tabra-ai/tabular-blends",
        organization="Tabra AI",
        license="StableAI LimiX Non-Commercial License v1.0; Kumo OpenMDW-1.1",
        commercial_use=False,
        tags=(),
        verified=False,
    ),
    pip_extra=(
        "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19",
        "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@a917fa052eb637db98318b65712def280055ec34",
        "nvtx",
    ),
)
