from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tababax.hpo import gen_tababax
from tabarena.models.tababax.model import TabAbaxModel

tababax_descriptor = ModelDescriptor(
    display_name="TabAbax",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/tababax-research/TabAbax",
    organization="TabAbax",
    commercial_use=False,
    license="CC-BY-NC-4.0 (weights); BSD-3-Clause for the TabICL-derived code",
    date_introduced="2026-10-28",
)

# Benchmarked, not yet hosted: TabArena v0.1 (816 splits) and BeyondArena core2k (2000 splits) are in
# the TabAbax repository's results/ folder. Register this entry in contexts/*/methods.py only once the
# processed artifacts are uploaded, so the collection resolves to downloadable results.
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
        "tababax @ git+https://github.com/tababax-research/TabAbax.git@9b7d916",
    ),
    prefetch_weights=TabAbaxModel.prefetch_weights,
)
