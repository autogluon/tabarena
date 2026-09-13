from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.aplr.hpo import gen_aplr
from tabarena.models.aplr.model import APLRModel

_aplr_common = {
    "compute": "cpu",
    "is_bag": True,
    "date_introduced": "2022",
    "reference_url": "https://github.com/ottenbreit-data-science/aplr",
    "verified": True,
}

aplr_method_metadata = MethodMetadata.config(
    method="aplr",
    suite="tabarena-2026-09-13",
    ag_key="TA-APLR",
    config_default="aplr_c1_default_BAG_L1",
    display_name="APLR",
    date="2026-09-13",
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    **_aplr_common,
)

aplr_info = ModelInfo(
    model_cls=APLRModel,
    search_space=gen_aplr,
    method_metadata=aplr_method_metadata,
    pip_extra=("aplr>=10.26.0",),
)
