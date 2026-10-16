from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabstar_v1p1.hpo import gen_tabstar_v1p1
from tabarena.models.tabstar_v1p1.model import TabSTARv1p1Model, prefetch_weights

tabstar_v1p1_method_metadata = MethodMetadata.config(
    method="TabSTAR-v1.1",
    ag_key="TA-TABSTAR-V1.1",
    can_hpo=False,
    compute="gpu",
    is_bag=True,
    date="2026-10-16",
    date_introduced="2026-10",
    reference_url="https://arxiv.org/abs/2505.18125",
    license="MIT (code); CC-BY-4.0 (weights)",
    display_name="TabSTAR-v1.1",
    # No benchmark run has been uploaded for this model yet.
    has_raw=False,
    has_processed=False,
    has_results=False,
    verified=False,
)


tabstar_v1p1_info = ModelInfo(
    model_cls=TabSTARv1p1Model,
    search_space=gen_tabstar_v1p1,
    method_metadata=tabstar_v1p1_method_metadata,
    pip_extra=("tabstar==1.2.0",),
    prefetch_weights=prefetch_weights,
)
