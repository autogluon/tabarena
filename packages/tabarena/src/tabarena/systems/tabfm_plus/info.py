from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models.tabfm.model import prefetch_weights
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.tabfm_plus.hpo import gen_tabfm_plus
from tabarena.systems.tabfm_plus.system import TabFMPlusSystemModel

# TabArena-Full run (816 tasks over 51 datasets), one direct fit of TabFM's `ensemble` interface
# per task with a 4 hour limit. `compute` is set by hand: the jobs requested one GPU each, but the
# raw results record `num_gpus=0`, so the inferred value would be `cpu`.
#
# The run predates the 2026-07-13 rerun that stopped reloading models from disk around inference,
# so its `time_infer` carries that reload and reads high next to the reruns. Hence `verified=False`
# until it is measured the same way.
tabfm_plus_method_metadata = MethodMetadata.system(
    method="TabFM+",
    name="TabFM+",
    suite="tabarena-2026-07-07",
    compute="gpu",
    date="2026-07-09",
    date_introduced="2026-06-30",
    reference_url="https://github.com/google-research/tabfm",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
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
