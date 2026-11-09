from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpack.hpo import gen_tabpack
from tabarena.models.tabpack.model import TabPackModel

# Shared by the TabArena entry below and the BeyondArena one in contexts/beyondarena/methods.py.
tabpack_descriptor = ModelDescriptor(
    display_name="TabPack",
    compute="gpu",
    is_bag=True,
    reference_url="https://arxiv.org/abs/2607.05380",
    date_introduced="2026-07-06",  # arXiv v1
    organization="Yandex Research",
    license="Apache-2.0",
)

# Run tabpack_08112026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md), default config only, at e2113666.
tabpack_method_metadata = tabpack_descriptor.method_metadata(
    method="TabPack",
    suite="tabarena-2026-11-08",
    ag_key="TA-TABPACK",
    model_key="TABPACK",
    config_default="TabPack_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-11-08",
    validation_protocol="8x1",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

tabpack_info = ModelInfo(
    model_cls=TabPackModel,
    search_space=gen_tabpack,
    method_metadata=tabpack_method_metadata,
    # tabpackage is experimental ("expect breaking changes"), and the wrapper moves the fitted
    # predictor between devices through two private attributes, so the release is pinned exactly.
    pip_extra=("tabpackage[sklearn]==0.0.2.dev0",),
)
