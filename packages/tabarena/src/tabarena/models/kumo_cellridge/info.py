from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.kumo_cellridge.hpo import gen_kumo_cellridge
from tabarena.models.kumo_cellridge.model import KumoCellRidgeModel

_descriptor = ModelDescriptor(
    display_name="Kumo-CellRidge-1024",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/1587causalai/kumo-small-cellridge-1024",
    organization="1587causalai",
    license="OpenMDW-1.1",
    date_introduced="2026-11-18",
)
kumo_cellridge_method_metadata = _descriptor.method_metadata(
    method="Kumo-CellRidge-1024",
    ag_key=KumoCellRidgeModel.ag_key,
    config_default="Kumo-CellRidge-1024_c1_default_BAG_L1",
    can_hpo=False,
    verified=False,
)
kumo_cellridge_info = ModelInfo(
    model_cls=KumoCellRidgeModel,
    search_space=gen_kumo_cellridge,
    method_metadata=kumo_cellridge_method_metadata,
    pip_extra=(
        "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@b780fc92b2cd50df5525575f0ddd1ae9038102bc",
        "safetensors>=0.4",
    ),
    prefetch_weights=KumoCellRidgeModel.prefetch_weights,
)
