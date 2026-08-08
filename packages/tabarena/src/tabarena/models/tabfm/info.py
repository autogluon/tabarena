from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabfm.hpo import gen_tabfm
from tabarena.models.tabfm.model import TabFMModel, prefetch_weights

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
tabfm_method_metadata = MethodMetadata.config(
    method="TabFM",
    suite="tabarena-2026-07-07",
    ag_key="TA-TABFM",
    model_key="TABFM",
    config_default="TabFM_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-07",
    date_introduced="2026-06-30",
    reference_url="https://github.com/google-research/tabfm",
    display_name="TabFM",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Pareto-front rerun with the improved train/infer time measurement.
tabfm_new_method_metadata = MethodMetadata.config(
    method="TabFM",
    suite="tabarena-2026-07-13",
    ag_key="TA-TABFM",
    model_key="TABFM",
    config_default="TabFM_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-15",
    date_introduced="2026-06-30",
    reference_url="https://github.com/google-research/tabfm",
    display_name="TabFM",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabfm_info = ModelInfo(
    model_cls=TabFMModel,
    search_space=gen_tabfm,
    method_metadata=tabfm_new_method_metadata,
    pip_extra=(
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@633cd265f498e1d20c9625be0639f6305d8e2541",
    ),
    prefetch_weights=prefetch_weights,
)

# TabFM+ (TabFM's heavier ``ensemble`` interface) shares this model's checkpoint but is
# benchmarked as a system, so it lives in `tabarena.systems.tabfm_plus`.
