from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.compacttab.hpo import gen_compacttab, gen_compacttab_hybrid
from tabarena.models.compacttab.model import CompactTabHybridModel, CompactTabModel

_PIP_EXTRA = ("compacttab @ git+https://github.com/DakshBhalala/TBDML.git@88107b3",)

compacttab_method_metadata = MethodMetadata.config(
    method="CompactTab",
    suite="tabarena-2026-10-20",
    ag_key="TA-COMPACTTAB",
    can_hpo=True,
    compute="gpu",
    is_bag=True,
    date="2026-10-20",
    date_introduced="2026-10",
    reference_url="https://github.com/DakshBhalala/TBDML",
    license="Apache-2.0",
    display_name="CompactTab",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    validation_protocol="8x1",
)

compacttab_hybrid_method_metadata = MethodMetadata.config(
    method="CompactTab-Hybrid",
    suite="tabarena-2026-10-20",
    ag_key="TA-COMPACTTAB-HYBRID",
    can_hpo=True,
    compute="gpu",
    is_bag=True,
    date="2026-10-20",
    date_introduced="2026-10",
    reference_url="https://github.com/DakshBhalala/TBDML",
    license="Apache-2.0",
    display_name="CompactTab-Hybrid",
    verified=False,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    validation_protocol="8x1",
)

compacttab_info = ModelInfo(
    model_cls=CompactTabModel,
    search_space=gen_compacttab,
    method_metadata=compacttab_method_metadata,
    pip_extra=_PIP_EXTRA,
)

compacttab_hybrid_info = ModelInfo(
    model_cls=CompactTabHybridModel,
    search_space=gen_compacttab_hybrid,
    method_metadata=compacttab_hybrid_method_metadata,
    pip_extra=_PIP_EXTRA,
)
