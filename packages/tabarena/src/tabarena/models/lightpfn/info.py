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

_run_kwargs = {
    "method": "LightPFN",
    "ag_key": "TA-LIGHTPFN",
    "model_key": "LIGHTPFN",
    "config_default": "LightPFN_c1_default_BAG_L1",
    "can_hpo": False,
    "validation_protocol": "8x1",
    "cache_type": "r2",
    "cache_kwargs": {"bucket": "tabarena", "prefix": "cache"},
}

# Run lightpfn15t214_10112026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md): all 816 splits, classification and
# regression, at c82b4dad, lightpfn 1.5.0, weights ueuegio/LightPFN@00a2536c, torch 2.14.1.
lightpfn_method_metadata = lightpfn_descriptor.method_metadata(
    **_run_kwargs,
    date="2026-11-10",
    suite="tabarena-2026-11-10",
    verified=True,
)

# Run lightpfn_09102026 (SkyPilot pool, 32 RTX PRO 6000, BENCHMARK_LOG.md): every split of the 38 classification
# datasets at 3d8217c0, lightpfn 1.0.0, weights ueuegio/LightPFN@bd389ab5. Superseded by the 1.5.0 run above.
lightpfn_1p0_method_metadata = lightpfn_descriptor.method_metadata(
    **_run_kwargs,
    date="2026-10-09",
    suite="tabarena-2026-10-09",
    verified=True,
)

lightpfn_info = ModelInfo(
    model_cls=LightPFNModel,
    search_space=gen_lightpfn,
    method_metadata=lightpfn_method_metadata,
    pip_extra=("lightpfn==1.5.0", "tabpfn-extensions[many_class]>=0.6.1"),
    prefetch_weights=prefetch_weights,
)
