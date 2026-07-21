from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.models._model_info import ModelInfo
from tabarena.models.nori.hpo import gen_nori, gen_nori30m
from tabarena.models.nori.model import Nori30MModel, NoriModel

# Superseded by the rerun below; kept so the hosted artifacts stay loadable.
nori_method_metadata = MethodMetadata.config(
    method="Nori",
    suite="tabarena-2026-06-30",
    ag_key="TA-NORI",
    config_default="Nori_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-06-18",
    date_introduced="2026-06-12",
    reference_url="https://github.com/Synthefy/synthefy-nori",
    display_name="Nori",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

# Pareto-front rerun with the improved train/infer time measurement.
nori_new_method_metadata = MethodMetadata.config(
    method="Nori",
    suite="tabarena-2026-07-13",
    ag_key="TA-NORI",
    config_default="Nori_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-15",
    date_introduced="2026-06-12",
    reference_url="https://github.com/Synthefy/synthefy-nori",
    display_name="Nori",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

nori_info = ModelInfo(
    model_cls=NoriModel,
    search_space=gen_nori,
    method_metadata=nori_new_method_metadata,
    pip_extra=("synthefy-nori>=0.7.0",),
    prefetch_weights=NoriModel.prefetch_weights,
)

# Nori-30M — a larger (~29M) version of Nori; a separate leaderboard entry (does not supersede
# the base above). Auto-discovered by the model registry alongside nori_info.
nori30m_method_metadata = MethodMetadata.config(
    method="Nori-30M",
    suite="tabarena-2026-07-13",
    ag_key="TA-NORI-30M",
    config_default="Nori-30M_c1_default_BAG_L1",
    can_hpo=False,
    compute="gpu",
    is_bag=False,
    date="2026-07-21",
    date_introduced="2026-07-21",
    reference_url="https://huggingface.co/Synthefy/Nori-30M",
    display_name="Nori-30M",
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

nori30m_info = ModelInfo(
    model_cls=Nori30MModel,
    search_space=gen_nori30m,
    method_metadata=nori30m_method_metadata,
    pip_extra=("synthefy-nori>=0.10.0",),  # >=0.10.0 provides the model= variant selector
    prefetch_weights=Nori30MModel.prefetch_weights,
)
