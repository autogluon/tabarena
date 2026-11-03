from __future__ import annotations

from tabarena.models._method_metadata import ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.seldon.hpo import gen_seldon, gen_seldon_mini
from tabarena.models.seldon.model import SeldonMiniModel, SeldonModel

#: Keep in sync with the `seldon` extra in pyproject.toml.
_PIP_EXTRA = ("neuralk-oss==0.2.3",)


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


# Runs seldonfix_03112026 and seldonminifix_03112026 (SkyPilot pools of 32 RTX PRO 6000, torch 2.13,
# BENCHMARK_LOG.md), the 38 classification datasets, every split, with the context cache kept out of the pickle.
_tabarena_run_kwargs = dict(
    can_hpo=False,
    date="2026-11-03",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    suite="tabarena-2026-11-03",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

seldon_method_metadata = _descriptor("Seldon").method_metadata(
    method="Seldon", ag_key="TA-SELDON", config_default="Seldon_c1_default_BAG_L1", **_tabarena_run_kwargs
)
seldon_mini_method_metadata = _descriptor("Seldon-Mini").method_metadata(
    method="Seldon-Mini",
    ag_key="TA-SELDON-MINI",
    config_default="Seldon-Mini_c1_default_BAG_L1",
    **_tabarena_run_kwargs,
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
