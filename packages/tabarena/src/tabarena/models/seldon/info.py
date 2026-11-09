from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata, ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.seldon.hpo import gen_seldon, gen_seldon_mini
from tabarena.models.seldon.model import SeldonMiniModel, SeldonModel

#: Keep in sync with the `seldon` extra in pyproject.toml.
_PIP_EXTRA = ("neuralk-oss==0.3.0",)


def _descriptor(display_name: str, reference_url: str) -> ModelDescriptor:
    return ModelDescriptor(
        display_name=display_name,
        compute="gpu",
        is_bag=False,
        reference_url=reference_url,
        organization="Neuralk-AI",
        # The weights' license: research and internal evaluation only (the neuralk-oss code is Apache-2.0).
        license="Seldon Research and Evaluation License",
        commercial_use=False,
        date_introduced="2026-11-02",
    )


# Classification: runs seldonfix_03112026 and seldonminifix_03112026, the 38 classification datasets. Regression: run
# seldonreg_08112026, the 13 regression datasets. Both on SkyPilot pools of 32 RTX PRO 6000 with torch 2.13, every
# split, with the context cache kept out of the pickle (BENCHMARK_LOG.md).
# The classification results come from neuralk-oss 0.2.3 with the checkpoints at Neuralk-AI/seldon@b113e1ce and
# Neuralk-AI/seldon-mini@28507308; the regression results (run seldonreg_08112026) come from the pins in `_PIP_EXTRA`
# and `model.HF_REVISIONS`. The classifier weights and packaged configs are byte-identical across the two, so the
# classification predictions are the same. Only timing differs: 0.3.0 caps the cached context at
# `max_cache_gb=8.0`, past which members encode their context again at every predict, so on the largest
# classification tables a 0.3.0 run reports more inference time and less fit time than the hosted results.
_tabarena_run_kwargs = dict(
    can_hpo=False,
    date="2026-11-08",
    verified=True,
    cache_type="r2",
    validation_protocol="8x1",
    # The classification results of the 2026-11-03 runs merged with the regression results of seldonreg_08112026:
    # all 51 datasets, every split.
    suite="tabarena-2026-11-08",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)
# The classification results alone, superseded by the suite above; kept so the hosted artifacts stay loadable.
_tabarena_classification_run_kwargs = {
    **_tabarena_run_kwargs,
    "date": "2026-11-03",
    "verified": True,
    "suite": "tabarena-2026-11-03",
}


def _method_metadata(method: str, ag_key: str, reference_url: str, run_kwargs: dict) -> MethodMetadata:
    return _descriptor(method, reference_url).method_metadata(
        method=method, ag_key=ag_key, config_default=f"{method}_c1_default_BAG_L1", **run_kwargs
    )


seldon_method_metadata = _method_metadata(
    "Seldon", "TA-SELDON", "https://huggingface.co/Neuralk-AI/seldon", _tabarena_run_kwargs
)
seldon_mini_method_metadata = _method_metadata(
    "Seldon-Mini", "TA-SELDON-MINI", "https://huggingface.co/Neuralk-AI/seldon-mini", _tabarena_run_kwargs
)
seldon_classification_method_metadata = _method_metadata(
    "Seldon", "TA-SELDON", "https://huggingface.co/Neuralk-AI/seldon", _tabarena_classification_run_kwargs
)
seldon_mini_classification_method_metadata = _method_metadata(
    "Seldon-Mini",
    "TA-SELDON-MINI",
    "https://huggingface.co/Neuralk-AI/seldon-mini",
    _tabarena_classification_run_kwargs,
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
