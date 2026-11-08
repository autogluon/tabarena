from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabh2o.hpo import gen_tabh2o
from tabarena.models.tabh2o.model import TabH2OModel, prefetch_weights

tabh2o_method_metadata = MethodMetadata.config(
    method="TabH2O",
    display_name="TabH2O",
    compute="gpu",
    date="2026-11-08",
    date_introduced="2026-06-01",
    ag_key="TA-TABH2O",
    model_key="TABH2O",
    can_hpo=False,
    is_bag=False,
    validation_protocol="8x1",
    suite="tabarena-2026-11-08",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    reference_url="https://huggingface.co/h2oai/tabh2o-v1.1",
    organization="H2O.ai",
    commercial_use=False,
    license="H2O Model Non-Commercial Licence v1.0 (weights)",
)

tabh2o_info = ModelInfo(
    model_cls=TabH2OModel,
    search_space=gen_tabh2o,
    method_metadata=tabh2o_method_metadata,
    pip_extra=("h2ofm==0.43.1",),
    prefetch_weights=prefetch_weights,
)
