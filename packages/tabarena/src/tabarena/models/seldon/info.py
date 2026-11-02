from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.seldon.hpo import gen_seldon, gen_seldon_mini
from tabarena.models.seldon.model import SeldonMiniModel, SeldonModel

#: Keep in sync with the `seldon` extra in pyproject.toml.
_PIP_EXTRA = ("neuralk-oss==0.2.1",)


def _descriptor(display_name: str) -> ModelDescriptor:
    return ModelDescriptor(
        display_name=display_name,
        compute="gpu",
        is_bag=False,
        reference_url="https://technical-report.neuralk.ai/",
        organization="Neuralk-AI",
        # The weights' license: research and internal evaluation only (the neuralk-oss code is Apache-2.0).
        license="Seldon Research and Evaluation License",
        commercial_use=False,
        date_introduced="2026-11-02",
    )


seldon_method_metadata = _descriptor("Seldon").method_metadata(
    method="Seldon", ag_key="TA-SELDON", can_hpo=False, date="2026-11-02", verified=False, suite="tabarena-2026-10-06"
)
seldon_mini_method_metadata = _descriptor("Seldon-Mini").method_metadata(
    method="Seldon-Mini",
    ag_key="TA-SELDON-MINI",
    can_hpo=False,
    date="2026-11-02",
    verified=False,
    suite="tabarena-2026-10-06",
)

seldon_info = ModelInfo(
    model_cls=SeldonModel,
    search_space=gen_seldon,
    method_metadata=seldon_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=SeldonModel.prefetch_weights,
)
seldon_mini_info = ModelInfo(
    model_cls=SeldonMiniModel,
    search_space=gen_seldon_mini,
    method_metadata=seldon_mini_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=SeldonMiniModel.prefetch_weights,
)
