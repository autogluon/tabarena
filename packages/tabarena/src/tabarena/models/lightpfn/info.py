from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.lightpfn.hpo import gen_lightpfn
from tabarena.models.lightpfn.model import LightPFNModel, prefetch_weights

#: Intrinsic facts shared by every run of the model.
lightpfn_descriptor = ModelDescriptor(
    display_name="LightPFN",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/GioOtto/LightPFN",
    license="Apache-2.0",
    commercial_use=True,
    date_introduced="2026-10-05",
)

lightpfn_method_metadata = lightpfn_descriptor.method_metadata(
    method="LightPFN",
    date="2026-10-08",
    ag_key="TA-LIGHTPFN",
    model_key="LIGHTPFN",
    can_hpo=False,
    validation_protocol="8x1",
    suite="tabarena-2026-10-08",
    verified=False,
)

lightpfn_info = ModelInfo(
    model_cls=LightPFNModel,
    search_space=gen_lightpfn,
    method_metadata=lightpfn_method_metadata,
    pip_extra=("lightpfn==1.0.0",),
    prefetch_weights=prefetch_weights,
)
