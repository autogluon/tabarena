from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.val_select.hpo import gen_val_select
from tabarena.systems.val_select.system import ValSelectSystemModel

# ValSelect: holdout validation selection between LimiX-2-1016 and Kumo-Tabular (large), refit of the
# winner on the full training data. Both members' weights are public; LimiX-2's license is
# non-commercial, so the system is non-commercial as a whole.
val_select_method_metadata = MethodMetadata.system(
    method="ValSelect",
    name="ValSelect",
    suite="tabarena-2026-11-19",
    compute="gpu",
    date="2026-11-19",
    date_introduced="2026-11",
    reference_url="https://github.com/ziyue246",
    organization="Independent",
    commercial_use=False,
    license="LimiX-2 Non-Commercial License + OpenMDW-1.1 (Kumo weights); wrapper code Apache-2.0",
    verified=False,
)

val_select_info = SystemInfo(
    system_cls=ValSelectSystemModel,
    config_generator=gen_val_select,
    method_metadata=val_select_method_metadata,
    pip_extra=(
        "LimiX @ git+https://github.com/limix-ldm-ai/LimiX.git@a917fa052eb637db98318b65712def280055ec34",
        "nvtx",
        "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19",
        # More than ten classes: LimiX's output coding (ManyClassClassifier).
        "tabpfn-extensions[many_class]>=0.6.1",
    ),
    prefetch_weights=None,
)
