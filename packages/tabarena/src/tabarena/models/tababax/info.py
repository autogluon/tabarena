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

# Run tababax_02112026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md): every split of the 51 TabArena-v0.1
# datasets at f47f00a0, tababax @ b7c9ba70, weights tababax-ai/TabAbax@039e4f23. The pin below, 125d2685, differs
# from b7c9ba70 only in setting HF_HUB_DISABLE_XET on Windows alone, which leaves the predictions unchanged.
tababax_method_metadata = tababax_descriptor.method_metadata(
    method="TabAbax",
    date="2026-11-02",
    ag_key="TA-TABABAX",
    model_key="TABABAX",
    config_default="TabAbax_c1_default_BAG_L1",
    can_hpo=False,
    validation_protocol="8x1",
    suite="tabarena-2026-11-02",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

tababax_info = ModelInfo(
    model_cls=TabAbaxModel,
    search_space=gen_tababax,
    method_metadata=tababax_method_metadata,
    pip_extra=(
        "tababax @ git+https://github.com/tababax-research/TabAbax.git@125d26850015597ed371993f3a00375806c0f32c",
    ),
    prefetch_weights=TabAbaxModel.prefetch_weights,
)
