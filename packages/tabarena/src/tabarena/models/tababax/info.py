from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tababax.hpo import gen_tababax
from tabarena.models.tababax.model import TabAbaxModel

tababax_descriptor = ModelDescriptor(
    display_name="TabAbax",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/soda-inria/tabicl",
    organization="TabAbax",
    commercial_use=False,
    license="BSD-3-Clause (code derived from TabICL), CC-BY-NC-4.0 weights",
    date_introduced="2026-10-28",
)

# Not benchmarked yet. This entry only registers the wrapper. Do not add it to
# contexts/tabarena/methods.py until a TabArena-Lite run exists.
tababax_method_metadata = tababax_descriptor.method_metadata(
    method="TabAbax",
    ag_key="TA-TABABAX",
    model_key="TABABAX",
    can_hpo=False,
    date="2026-10-28",
)

tababax_info = ModelInfo(
    model_cls=TabAbaxModel,
    search_space=gen_tababax,
    method_metadata=tababax_method_metadata,
    pip_extra=(
        "tababax @ git+https://github.com/tababax-research/TabAbax.git@0ff3c22d0f824444d7493118db5b9fc7167298c6",
    ),
)
