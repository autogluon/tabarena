from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.inpriora.hpo import gen_inpriora
from tabarena.models.inpriora.model import InprioraModel, prefetch_weights

#: Keep in sync with packages/tabarena/pyproject.toml. The submitted CUDA run used Triton.
_PIP_EXTRA = (
    "inpriora[triton] @ git+https://github.com/ai-for-data/inpriora.git@dde547d0559ee825a5e3830825c2bfff9badc6ab",
)

#: Intrinsic facts shared by every run of the model.
inpriora_descriptor = ModelDescriptor(
    display_name="Inpriora",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/ai-for-data/inpriora",
    license="Apache-2.0",
    commercial_use=True,
    organization="aifordata",
    date_introduced="2026-11",
)

inpriora_method_metadata = inpriora_descriptor.method_metadata(
    method="Inpriora",
    ag_key="TA-INPRIORA",
    model_key="INPRIORA",
    can_hpo=False,
    validation_protocol="8x1",
    suite="tabarena-2026-11-12",
    date="2026-11-12",
    verified=False,
)

inpriora_info = ModelInfo(
    model_cls=InprioraModel,
    search_space=gen_inpriora,
    method_metadata=inpriora_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=prefetch_weights,
)
