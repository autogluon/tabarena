from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.kumo_tabular.hpo import gen_kumo_tabular, gen_kumo_tabular_medium, gen_kumo_tabular_small
from tabarena.models.kumo_tabular.model import KumoTabularMediumModel, KumoTabularModel, KumoTabularSmallModel

#: Not on PyPI yet (only a placeholder release); pinned to a commit so the benchmarked code is fixed.
#: Keep in sync with the `kumo_tabular` extra in pyproject.toml.
_PIP_EXTRA = (
    "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@98f61289c7a4e1bce3b33771223ac2e123b63f19",
)


def _descriptor(display_name: str) -> ModelDescriptor:
    return ModelDescriptor(
        display_name=display_name,
        compute="gpu",
        is_bag=False,
        reference_url="https://huggingface.co/blog/nvidia/kumo-tabular",
        license="OpenMDW-1.1",
        date_introduced="2026-09-25",  # the Hub release commit of the v1.0.0 checkpoints
    )


kumo_tabular_descriptor = _descriptor("Kumo-Tabular")
kumo_tabular_medium_descriptor = _descriptor("Kumo-Tabular-Medium")
kumo_tabular_small_descriptor = _descriptor("Kumo-Tabular-Small")

# Run kumotabular_06102026 (SkyPilot pool, 64 RTX PRO 6000, BENCHMARK_LOG.md), all three sizes in one run, at a98c3381:
# the context KV cache only for the refit model (#641), saved beside model.pkl (#644), sdm 98f61289, Hub tag v1.0.1.
_tabarena_run_kwargs = dict(
    can_hpo=False,
    date="2026-10-06",
    verified=False,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-10-06",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

kumo_tabular_method_metadata = kumo_tabular_descriptor.method_metadata(
    method="Kumo-Tabular",
    ag_key="TA-KUMO-TABULAR",
    config_default="Kumo-Tabular_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
)

kumo_tabular_medium_method_metadata = kumo_tabular_medium_descriptor.method_metadata(
    method="Kumo-Tabular-Medium",
    ag_key="TA-KUMO-TABULAR-MEDIUM",
    config_default="Kumo-Tabular-Medium_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
)

kumo_tabular_small_method_metadata = kumo_tabular_small_descriptor.method_metadata(
    method="Kumo-Tabular-Small",
    ag_key="TA-KUMO-TABULAR-SMALL",
    config_default="Kumo-Tabular-Small_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
)


kumo_tabular_info = ModelInfo(
    model_cls=KumoTabularModel,
    search_space=gen_kumo_tabular,
    method_metadata=kumo_tabular_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=KumoTabularModel.prefetch_weights,
)

kumo_tabular_medium_info = ModelInfo(
    model_cls=KumoTabularMediumModel,
    search_space=gen_kumo_tabular_medium,
    method_metadata=kumo_tabular_medium_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=KumoTabularMediumModel.prefetch_weights,
)

kumo_tabular_small_info = ModelInfo(
    model_cls=KumoTabularSmallModel,
    search_space=gen_kumo_tabular_small,
    method_metadata=kumo_tabular_small_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=KumoTabularSmallModel.prefetch_weights,
)
