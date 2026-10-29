from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tababax.hpo import gen_tababax
from tabarena.models.tababax.model import TababaxModel

tababax_descriptor = ModelDescriptor(
    display_name="Tababax",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/tababax-research/Tababax",
    organization="Tababax",
    commercial_use=False,
    license="Apache-2.0 code, CC-BY-NC-4.0 weights",
    date_introduced="2026-10-28",
)

# Not benchmarked yet. This entry only registers the wrapper. Do not add it to
# contexts/tabarena/methods.py until a TabArena-Lite run exists.
tababax_method_metadata = tababax_descriptor.method_metadata(
    method="Tababax",
    ag_key="TA-TABABAX",
    model_key="TABABAX",
    can_hpo=False,
    date="2026-10-28",
)

tababax_info = ModelInfo(
    model_cls=TababaxModel,
    search_space=gen_tababax,
    method_metadata=tababax_method_metadata,
    pip_extra=(
        "tababax @ git+https://github.com/tababax-research/Tababax.git@1a90fa8c11e29e54d91e8807fe152ab751693863",
    ),
)
