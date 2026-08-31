from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabldm.hpo import gen_tabldm
from tabarena.models.tabldm.model import TabLDMModel

# Not yet benchmarked: no `suite`/`cache_kwargs` (local, unhosted), `has_raw`/`has_processed`/
# `has_results` all False since no artifacts exist yet, `date` is the planning date below.
tabldm_method_metadata = MethodMetadata.config(
    method="TabLDM",
    display_name="TabLDM",
    compute="gpu",
    date="2026-08-31",
    ag_key="TA-TABLDM",
    model_key="TABLDM",
    config_default="TabLDM_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=False,
    has_processed=False,
    has_results=False,
    verified=False,
    reference_url="https://huggingface.co/occams/Xiaomi-TabLDM",
)


tabldm_info = ModelInfo(
    model_cls=TabLDMModel,
    search_space=gen_tabldm,
    method_metadata=tabldm_method_metadata,
    # Vendored under `_vendor/` (not on PyPI). Most runtime deps (torch, numpy, scikit-learn,
    # scipy, psutil, tqdm, huggingface_hub) are already in TabArena's base tree, but `einops`
    # (used by `_model/rope.py`) is not, so it is the one real extra dependency.
    pip_extra=("einops",),
    prefetch_weights=TabLDMModel.prefetch_weights,
)
