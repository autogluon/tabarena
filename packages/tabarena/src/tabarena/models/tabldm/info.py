from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabldm.hpo import gen_tabldm
from tabarena.models.tabldm.model import TabLDMModel

# Not yet benchmarked: no `suite`/`cache_kwargs` (local, unhosted), `has_raw`/`has_processed`/
# `has_results` all False since no artifacts exist yet, `date` is the planning date below.
tabldm_method_metadata = MethodMetadata.config(
    method="Xiaomi-TabLDM",
    display_name="Xiaomi-TabLDM",
    compute="gpu",
    date="2026-08-31",
    ag_key="TA-XIAOMI-TABLDM",
    model_key="XIAOMI-TABLDM",
    config_default="Xiaomi-TabLDM_c1_BAG_L1",
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
    # TabLDM is not published on PyPI; pinned to a commit so the benchmarked code is fixed.
    # Keep in sync with the `tabldm` extra in pyproject.toml.
    pip_extra=(
        "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@3090d4f3da420e25a482bead97c9fe061607bffb",
    ),
    prefetch_weights=TabLDMModel.prefetch_weights,
)
