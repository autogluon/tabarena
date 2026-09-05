from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.aplr.hpo import gen_aplr_deep_int, gen_aplr_two_way_int
from tabarena.models.aplr.model import APLRDeepIntModel, APLRTwoWayIntModel

_aplr_common = {
    "compute": "cpu",
    "is_bag": True,
    "date_introduced": "2022",
    "reference_url": "https://github.com/ottenbreit-data-science/aplr",
    "verified": False,
}

aplr_two_way_int_method_metadata = MethodMetadata.config(
    method="aplr_two_way_int",
    suite="tabarena-2026-09-04",
    ag_key="TA-APLR_TWO_WAY_INT",
    config_default="aplr_two_way_int_c1_BAG_L1",
    display_name="APLR (two-way interactions)",
    **_aplr_common,
)

aplr_deep_int_method_metadata = MethodMetadata.config(
    method="aplr_deep_int",
    suite="tabarena-2026-09-04",
    ag_key="TA-APLR_DEEP_INT",
    config_default="aplr_deep_int_c1_BAG_L1",
    display_name="APLR (deep interactions)",
    **_aplr_common,
)

aplr_two_way_int_info = ModelInfo(
    model_cls=APLRTwoWayIntModel,
    search_space=gen_aplr_two_way_int,
    method_metadata=aplr_two_way_int_method_metadata,
    pip_extra=("aplr>=10.26.0",),
)

aplr_deep_int_info = ModelInfo(
    model_cls=APLRDeepIntModel,
    search_space=gen_aplr_deep_int,
    method_metadata=aplr_deep_int_method_metadata,
    pip_extra=("aplr>=10.26.0",),
)
