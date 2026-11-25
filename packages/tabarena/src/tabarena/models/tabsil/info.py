from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabsil.hpo import gen_tabsil
from tabarena.models.tabsil.model import TabSilModel

#: Keep in sync with the `tabsil` extra in pyproject.toml.
_PIP_EXTRA = ("tabsil==0.1.0",)

tabsil_descriptor = ModelDescriptor(
    display_name="TabSil",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/Aiden-Jeon/tabsil",
    organization="Aiden-Jeon",
    license="Apache-2.0",
    date_introduced="2026-11-24",  # the Hub upload of the v1 weights
)

# Not benchmarked by the maintainers yet: no hosted results, so the cache is local.
tabsil_method_metadata = tabsil_descriptor.method_metadata(
    method="TabSil",
    ag_key="TA-TABSIL",
    can_hpo=False,
    date="2026-11-25",
    validation_protocol="8x1",
    suite="tabarena-2026-11-25",
)

tabsil_info = ModelInfo(
    model_cls=TabSilModel,
    search_space=gen_tabsil,
    method_metadata=tabsil_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=TabSilModel.prefetch_weights,
)
