from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.tabldm.hpo import gen_tabldm
from tabarena.models.tabldm.model import TabLDMModel

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
tabldm_method_metadata = MethodMetadata.config(
    method="Xiaomi-TabLDM",
    display_name="Xiaomi-TabLDM",
    compute="gpu",
    date="2026-09-11",
    # ag_key matches the run's raw data; model_key stays "XIAOMI-TABLDM" for stable registry naming.
    ag_key="TA-XIAOMI-TABLDM",
    model_key="XIAOMI-TABLDM",
    config_default="Xiaomi-TabLDM_c1_default_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    suite="tabarena-2026-09-11",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    reference_url="https://huggingface.co/occams/Xiaomi-TabLDM",
    license="Apache-2.0",
)

# Rerun on the timing and warm-up pipeline (PR #584).
tabldm_new_method_metadata = MethodMetadata.config(
    method="Xiaomi-TabLDM",
    display_name="Xiaomi-TabLDM",
    compute="gpu",
    date="2026-09-16",
    # ag_key matches the run's raw data; model_key stays "XIAOMI-TABLDM" for stable registry naming.
    ag_key="TA-XIAOMI-TABLDM",
    model_key="XIAOMI-TABLDM",
    config_default="Xiaomi-TabLDM_c1_default_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    validation_protocol="8x1",
    suite="tabarena-2026-09-16",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    reference_url="https://huggingface.co/occams/Xiaomi-TabLDM",
    license="Apache-2.0",
)


tabldm_info = ModelInfo(
    model_cls=TabLDMModel,
    search_space=gen_tabldm,
    method_metadata=tabldm_new_method_metadata,
    # TabLDM is not published on PyPI; pinned to a commit so the benchmarked code is fixed.
    # Keep in sync with the `tabldm` extra in pyproject.toml.
    pip_extra=(
        "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@6773a30d43e43fad3e8b474e20ca8c7ec40dcd76",
    ),
    prefetch_weights=TabLDMModel.prefetch_weights,
)
