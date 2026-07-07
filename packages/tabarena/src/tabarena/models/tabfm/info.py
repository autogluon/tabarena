from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabfm.hpo import gen_tabfm
from tabarena.models.tabfm.model import TabFMModel, prefetch_weights

tabfm_method_metadata = MethodMetadata.config(
    method="TabFM",
    suite="tabarena-2026-06-26",
    ag_key="TA-TABFM",
    model_key="TABFM",
    config_default="TabFM_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-06-26",
    reference_url="https://github.com/google-research/tabfm",
    display_name="TabFM",
    verified=False,
)


tabfm_info = ModelInfo(
    model_cls=TabFMModel,
    search_space=gen_tabfm,
    method_metadata=tabfm_method_metadata,
    pip_extra=(
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@633cd265f498e1d20c9625be0639f6305d8e2541",
    ),
    prefetch_weights=prefetch_weights,
)


# TabFM+ (TabFM's ``ensemble`` interface) is benchmarked as a system, not an AutoGluon model, so it
# has its own MethodMetadata but no `ModelInfo`: it is not part of the auto-discovered model registry
# (`discover_models`) — it is wired into a benchmark run via `gen_tabfm_plus` (a
# `SystemConfigGenerator`) and the experiment bundle's `system_experiments=True` mode. A system's
# results are recorded as a `baseline` method (see `get_info_from_result`).
tabfm_plus_method_metadata = MethodMetadata.baseline(
    method="TabFM+",
    name="TabFM+",
    suite="tabarena-2026-06-26",
    compute="gpu",
    date="2026-06-26",
    reference_url="https://github.com/google-research/tabfm",
    verified=False,
)
