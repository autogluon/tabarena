from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.lgbm_prep_fte.hpo import gen_lgbm_prep_fte25, gen_lgbm_prep_fte200
from tabarena.systems.lgbm_prep_fte.system import NAMES, LGBMPrepFTESystemModel

# TODO(results): set `verified` and add `cache_type` / `cache_kwargs` per the upload-method skill once the results
# are hosted, then register both in the arenas' method collections.
_common_kwargs = dict(
    compute="cpu",
    date_introduced="2026-09",
    reference_url="https://github.com/atschalz/tabprep",
    organization="Yandex Research",
    license="Apache-2.0",
    commercial_use=True,
    tags=(),
    verified=False,
)

#: TabArena-v0.1 (all 816 splits): LightGBM default + the 200-config portfolio.
lgbm_prep_fte200_method_metadata = MethodMetadata.system(
    method=NAMES["tabarena"],
    name=NAMES["tabarena"],
    suite="tabarena-2026-11-20",
    date="2026-11-20",
    **_common_kwargs,
)

#: BeyondArena core2k (2,000 splits): LightGBM default + the 25-config portfolio.
lgbm_prep_fte25_method_metadata = MethodMetadata.system(
    method=NAMES["beyondarena"],
    name=NAMES["beyondarena"],
    suite="beyondarena-2026-11-20",
    date="2026-11-20",
    **_common_kwargs,
)

# LightGBM comes with TabArena's AutoGluon install; TabPrep's generators are part of autogluon.features.
lgbm_prep_fte200_info = SystemInfo(
    system_cls=LGBMPrepFTESystemModel,
    config_generator=gen_lgbm_prep_fte200,
    method_metadata=lgbm_prep_fte200_method_metadata,
    pip_extra=(),
)
lgbm_prep_fte25_info = SystemInfo(
    system_cls=LGBMPrepFTESystemModel,
    config_generator=gen_lgbm_prep_fte25,
    method_metadata=lgbm_prep_fte25_method_metadata,
    pip_extra=(),
)
