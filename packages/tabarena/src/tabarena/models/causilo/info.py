from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.causilo.hpo import gen_causilo
from tabarena.models.causilo.model import CausiloModel, prefetch_weights

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
causilo_method_metadata = MethodMetadata.config(
    method="Causilo",
    display_name="Causilo",
    compute="gpu",
    date="2026-09-13",
    date_introduced="2026-09-13",
    # ag_key matches the run's raw data; model_key stays "CAUSILO" for stable registry naming.
    ag_key="TA-CAUSILO",
    model_key="CAUSILO",
    config_default="Causilo_c1_default_BAG_L1",
    can_hpo=False,
    is_bag=False,
    has_raw=True,
    has_processed=True,
    has_results=True,
    suite="tabarena-2026-09-13",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
    reference_url="https://github.com/nums-ai/causilo",
    commercial_use=False,
    license="Causilo License v1.0 (non-commercial weights)",
)

# Rerun on the timing and warm-up pipeline (PR #584).
causilo_new_method_metadata = MethodMetadata.config(
    method="Causilo",
    display_name="Causilo",
    compute="gpu",
    date="2026-09-16",
    date_introduced="2026-09-13",
    # ag_key matches the run's raw data; model_key stays "CAUSILO" for stable registry naming.
    ag_key="TA-CAUSILO",
    model_key="CAUSILO",
    config_default="Causilo_c1_default_BAG_L1",
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
    reference_url="https://github.com/nums-ai/causilo",
    commercial_use=False,
    license="Causilo License v1.0 (non-commercial weights)",
)

causilo_info = ModelInfo(
    model_cls=CausiloModel,
    search_space=gen_causilo,
    method_metadata=causilo_new_method_metadata,
    pip_extra=("causilo==1.0.2",),
    prefetch_weights=prefetch_weights,
)
