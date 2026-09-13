from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.causilo.hpo import gen_causilo
from tabarena.models.causilo.model import CausiloModel, prefetch_weights

causilo_method_metadata = MethodMetadata.config(
    method="Causilo",
    display_name="Causilo",
    compute="gpu",
    ag_key="TA-CAUSILO",
    model_key="CAUSILO",
    config_default="Causilo_c1_default_BAG_L1",
    can_hpo=False,
    is_bag=False,
    suite="causilo-submission-2026-09-12",
    date="2026-09-12",
    has_raw=False,
    has_processed=False,
    has_results=False,
    verified=False,
    reference_url="https://github.com/nums-ai/causilo",
)

causilo_info = ModelInfo(
    model_cls=CausiloModel,
    search_space=gen_causilo,
    method_metadata=causilo_method_metadata,
    pip_extra=("causilo==1.0.0",),
    prefetch_weights=prefetch_weights,
)
