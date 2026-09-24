from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata, ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabfm.hpo import gen_tabfm
from tabarena.models.tabfm.model import TabFMModel, prefetch_weights

#: Intrinsic facts shared by every run of the model (the BeyondArena entries build on it).
tabfm_descriptor = ModelDescriptor(
    display_name="TabFM",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/google-research/tabfm",
    commercial_use=False,
    license="TabFM Non-Commercial License v1.0",
    date_introduced="2026-06-30",
)

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
    commercial_use=False,
    license="TabFM Non-Commercial License v1.0",
    display_name="TabFM",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Pareto-front rerun with the improved train/infer time measurement; superseded by the rerun below.
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
    commercial_use=False,
    license="TabFM Non-Commercial License v1.0",
    display_name="TabFM",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Rerun on the timing and warm-up pipeline (PR #584).
tabfm_2026_09_method_metadata = MethodMetadata.config(
    method="TabFM",
    validation_protocol="8x1",
    suite="tabarena-2026-09-16",
    ag_key="TA-TABFM",
    model_key="TABFM",
    config_default="TabFM_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-09-16",
    date_introduced="2026-06-30",
    reference_url="https://github.com/google-research/tabfm",
    display_name="TabFM",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    commercial_use=False,
    license="TabFM Non-Commercial License v1.0",
)


tabfm_info = ModelInfo(
    model_cls=TabFMModel,
    search_space=gen_tabfm,
    method_metadata=tabfm_2026_09_method_metadata,
    pip_extra=(
        "tabfm[pytorch] @ git+https://github.com/google-research/tabfm.git@fbb665569425fd2f490c6576b3af967876fe11ff",
        "tabpfn-extensions[many_class]>=0.6.1",
    ),
    prefetch_weights=prefetch_weights,
)

# TabFM+ (TabFM's heavier ``ensemble`` interface) shares this model's checkpoint but is
# benchmarked as a system, so it lives in `tabarena.systems.tabfm_plus`.
