from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.denselight.hpo import gen_denselight
from tabarena.models.denselight.model import DenseLightModel

# Not yet benchmarked in TabArena: has_raw / has_processed / has_results stay False and `verified`
# stays False until a real run is registered (see the add-model skill, "Metadata artifact" step).
denselight_method_metadata = MethodMetadata(
    method="DenseLight",
    method_type="config",
    display_name="DenseLight",
    compute="gpu",
    date="2026-06-22",
    ag_key="TA-DENSELIGHT",
    model_key="DENSELIGHT",
    config_default="DenseLight_c1_BAG_L1",
    can_hpo=True,
    is_bag=True,
    has_raw=False,
    has_processed=False,
    has_results=False,
    verified=False,
    reference_url="https://github.com/sb-ai-lab/LightAutoML",
    name_suffix=None,
)


denselight_info = ModelInfo(
    model_cls=DenseLightModel,
    search_space=gen_denselight,
    method_metadata=denselight_method_metadata,
    pip_extra=("lightautoml",),
)
