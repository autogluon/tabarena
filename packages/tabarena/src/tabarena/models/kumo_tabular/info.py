from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.kumo_tabular.hpo import gen_kumo_tabular, gen_kumo_tabular_medium, gen_kumo_tabular_small
from tabarena.models.kumo_tabular.model import KumoTabularMediumModel, KumoTabularModel, KumoTabularSmallModel

#: Not on PyPI yet (only a placeholder release); pinned to a commit so the benchmarked code is fixed.
#: Keep in sync with the `kumo_tabular` extra in pyproject.toml.
_PIP_EXTRA = (
    "structured-data-models @ git+https://github.com/NVIDIA/structured-data-models.git@fff8a503eed230868b4c11011fa4a7a9d2f78cc0",
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


kumo_tabular_method_metadata = _descriptor("Kumo-Tabular").method_metadata(
    method="Kumo-Tabular",
    ag_key="TA-KUMO-TABULAR",
    config_default="Kumo-Tabular_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-28",
)

kumo_tabular_medium_method_metadata = _descriptor("Kumo-Tabular-Medium").method_metadata(
    method="Kumo-Tabular-Medium",
    ag_key="TA-KUMO-TABULAR-MEDIUM",
    config_default="Kumo-Tabular-Medium_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-28",
)

kumo_tabular_small_method_metadata = _descriptor("Kumo-Tabular-Small").method_metadata(
    method="Kumo-Tabular-Small",
    ag_key="TA-KUMO-TABULAR-SMALL",
    config_default="Kumo-Tabular-Small_c1_default_BAG_L1",
    can_hpo=False,
    date="2026-09-28",
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
