from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.lgbm_prep_fte.hpo import gen_lgbm_prep_fte
from tabarena.systems.lgbm_prep_fte.system import NAME, LGBMPrepFTESystemModel

# TODO(results): set `suite`, `date` and `verified` once the TabArena-v0.1 and BeyondArena core2k results are
# hosted; add `cache_type` / `cache_kwargs` per the upload-method skill.
lgbm_prep_fte_method_metadata = MethodMetadata.system(
    method=NAME,
    name=NAME,
    suite="tabarena-2026-11-18",
    compute="cpu",
    date="2026-11-18",
    date_introduced="2026-09",
    reference_url="https://github.com/atschalz/tabprep",
    organization="Yandex Research",
    license="Apache-2.0",
    commercial_use=True,
    tags=(),
    verified=False,
)

lgbm_prep_fte_info = SystemInfo(
    system_cls=LGBMPrepFTESystemModel,
    config_generator=gen_lgbm_prep_fte,
    method_metadata=lgbm_prep_fte_method_metadata,
    # LightGBM comes with TabArena's AutoGluon install; TabPrep's generators are part of autogluon.features.
    pip_extra=(),
)
