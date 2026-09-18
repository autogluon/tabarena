from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpfn_3_5.hpo import gen_tabpfn_3_5, gen_tabpfn_3_5_fast
from tabarena.models.tabpfn_3_5.model import TabPFN35FastModel, TabPFN35Model

tabpfn_3_5_descriptor = ModelDescriptor(
    display_name="TabPFN-3.5",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2609.17895",
    commercial_use=False,
    license="TabPFN-3.5 License v1.0 (non-commercial)",
    date_introduced="2026-09-15",
)

tabpfn_3_5_fast_descriptor = ModelDescriptor(
    display_name="TabPFN-3.5-Fast",
    compute="gpu",
    is_bag=False,
    reference_url="https://arxiv.org/abs/2609.17895",
    commercial_use=False,
    license="TabPFN-3.5 License v1.0 (non-commercial)",
    date_introduced="2026-09-15",
)

# Run rerun_tabpfn35_17092026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md), both models in one run.
# It replaces suite tabarena-2026-09-17 (run tabpfn35_17092026), whose fit times carried the 15 to 20 s
# start-up of the preprocessing worker pool; those artifacts were deleted from r2.
tabpfn_3_5_method_metadata = tabpfn_3_5_descriptor.method_metadata(
    method="TabPFN-3.5",
    ag_key="TA-TABPFN-3.5",
    config_default="TabPFN-3.5_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-17",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-09-17-fix",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

tabpfn_3_5_fast_method_metadata = tabpfn_3_5_fast_descriptor.method_metadata(
    method="TabPFN-3.5-Fast",
    ag_key="TA-TABPFN-3.5-FAST",
    config_default="TabPFN-3.5-Fast_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-17",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-09-17-fix",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabpfn_3_5_info = ModelInfo(
    model_cls=TabPFN35Model,
    search_space=gen_tabpfn_3_5,
    method_metadata=tabpfn_3_5_method_metadata,
    pip_extra=("tabpfn>=9.0.0",),
    prefetch_weights=TabPFN35Model.prefetch_weights,
)

tabpfn_3_5_fast_info = ModelInfo(
    model_cls=TabPFN35FastModel,
    search_space=gen_tabpfn_3_5_fast,
    method_metadata=tabpfn_3_5_fast_method_metadata,
    pip_extra=("tabpfn>=9.0.0",),
    prefetch_weights=TabPFN35FastModel.prefetch_weights,
)
