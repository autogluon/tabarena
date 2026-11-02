from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabpack.hpo import gen_tabpack
from tabarena.models.tabpack.model import TabPackModel

# Not benchmarked yet: the suite, date, cache location and verification follow the run.
tabpack_method_metadata = MethodMetadata.config(
    method="TabPack",
    display_name="TabPack",
    ag_key="TA-TABPACK",
    model_key="TABPACK",
    can_hpo=False,
    compute="gpu",
    is_bag=True,
    date="2026-10-31",
    date_introduced="2026-07-06",  # arXiv v1
    reference_url="https://arxiv.org/abs/2607.05380",
    organization="Yandex Research",
    license="Apache-2.0",
    verified=False,
)

tabpack_info = ModelInfo(
    model_cls=TabPackModel,
    search_space=gen_tabpack,
    method_metadata=tabpack_method_metadata,
    # tabpackage is experimental ("expect breaking changes"), and the wrapper moves the fitted
    # predictor between devices through two private attributes, so the release is pinned exactly.
    pip_extra=("tabpackage[sklearn]==0.0.1",),
)
