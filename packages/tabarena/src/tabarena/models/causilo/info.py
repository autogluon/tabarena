from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata, ModelDescriptor
from tabarena.models._model_info import ModelInfo
from tabarena.models.causilo.hpo import gen_causilo, gen_causilo15
from tabarena.models.causilo.model import Causilo15Model, CausiloModel

#: Keep in sync with packages/tabarena/pyproject.toml. One release serves both checkpoints; its 1.0 runtime
#: reproduces the predictions of causilo 1.0.2 (the release of the hosted Causilo runs) exactly on the CPU.
_PIP_EXTRA = ("causilo==1.5.1",)

#: Intrinsic facts shared by every run of the model (the BeyondArena entries build on it).
causilo_descriptor = ModelDescriptor(
    display_name="Causilo",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/nums-ai/causilo",
    organization="Nums AI",
    commercial_use=False,
    license="Causilo License v1.0 (non-commercial weights)",
    date_introduced="2026-09-13",
)

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
    organization="Nums AI",
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
    organization="Nums AI",
    commercial_use=False,
    license="Causilo License v1.0 (non-commercial weights)",
)

causilo_info = ModelInfo(
    model_cls=CausiloModel,
    search_space=gen_causilo,
    method_metadata=causilo_new_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=CausiloModel.prefetch_weights,
)

causilo15_descriptor = ModelDescriptor(
    display_name="Causilo 1.5",
    compute="gpu",
    is_bag=False,
    reference_url="https://github.com/nums-ai/causilo",
    organization="Nums AI",
    commercial_use=False,
    license="Nums AI Model License (non-commercial weights)",
    date_introduced="2026-11-25",
)

# Run causilo15_25112026: classification with causilo 1.5.1 (4 members, 2026-11-26), regression with 1.5.0
# (8 members, 2026-11-25), which predicts bit-identically to 1.5.1 at 8 members.
causilo15_method_metadata = causilo15_descriptor.method_metadata(
    method="Causilo-1.5",
    date="2026-11-26",
    ag_key=Causilo15Model.ag_key,
    model_key="CAUSILO-1.5",
    config_default="Causilo-1.5_c1_default_BAG_L1",
    can_hpo=False,
    validation_protocol="8x1",
    suite="tabarena-2026-11-26",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

causilo15_info = ModelInfo(
    model_cls=Causilo15Model,
    search_space=gen_causilo15,
    method_metadata=causilo15_method_metadata,
    pip_extra=_PIP_EXTRA,
    prefetch_weights=Causilo15Model.prefetch_weights,
)
