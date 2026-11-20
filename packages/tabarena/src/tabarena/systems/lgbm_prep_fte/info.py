from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.lgbm_prep_fte.hpo import gen_lgbm_prep_fte
from tabarena.systems.lgbm_prep_fte.system import NAME, LGBMPrepFTESystemModel

# TODO(results): set `verified` and add `cache_type` / `cache_kwargs` per the upload-method skill once the results
# are hosted, then register both in the arenas' method collections.
_common_kwargs = dict(
    method=NAME,
    name=NAME,
    compute="cpu",
    date_introduced="2026-09",
    reference_url="https://github.com/atschalz/tabprep",
    organization="Yandex Research",
    license="Apache-2.0",
    commercial_use=True,
    tags=(),
    verified=False,
)

#: TabArena-v0.1 (all 816 splits): the 201-config portfolio.
lgbm_prep_fte_method_metadata = MethodMetadata.system(suite="tabarena-2026-11-20", date="2026-11-20", **_common_kwargs)

#: BeyondArena core2k (2,000 splits): the 26-config portfolio.
lgbm_prep_fte_beyondarena_method_metadata = MethodMetadata.system(
    suite="beyondarena-2026-11-20", date="2026-11-20", **_common_kwargs
)

lgbm_prep_fte_info = SystemInfo(
    system_cls=LGBMPrepFTESystemModel,
    config_generator=gen_lgbm_prep_fte,
    method_metadata=lgbm_prep_fte_method_metadata,
    # LightGBM comes with TabArena's AutoGluon install; TabPrep's generators are part of autogluon.features.
    pip_extra=(),
)
