from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.tabldm_plus.hpo import gen_tabldm_plus
from tabarena.systems.tabldm_plus.system import TabLDMPlusSystemModel, prefetch_weights

# Xiaomi-TabLDM's enhanced inference path, benchmarked as a system. The estimator fits its
# candidate/ensemble weights on a hold-out split it carves out of the training data, which the
# model protocol does not allow (TabArena passes a model its own validation split and a wrapper
# may not carve a second one) but `ExternalSystemModel` explicitly does. Hence a system entrant
# alongside the `Xiaomi-TabLDM` model entry rather than a new model version of it.
#
# `suite` and `date` describe the run this entry will carry and must be set to the actual run's
# date before the results are processed and uploaded (see the `upload-method` skill).
tabldm_plus_method_metadata = MethodMetadata.system(
    method="Xiaomi-TabLDM+",
    name="Xiaomi-TabLDM+",
    suite="tabarena-2026-10-08",
    compute="gpu",
    date="2026-10-08",
    date_introduced="2026-10",
    reference_url="https://huggingface.co/occams/Xiaomi-TabLDM",
    commercial_use=True,
    license="Apache-2.0",
    tags=(),
    verified=False,
)


tabldm_plus_info = SystemInfo(
    system_cls=TabLDMPlusSystemModel,
    config_generator=gen_tabldm_plus,
    method_metadata=tabldm_plus_method_metadata,
    # TabLDM is not published on PyPI; pinned to a commit so the benchmarked code is fixed.
    # Keep in sync with the `tabldm_plus` extra in pyproject.toml. This is a *different* commit
    # from the `tabldm` extra the Xiaomi-TabLDM model entry pins: the two are API-incompatible
    # (`TabLDMEnhanced*` vs `TabLDM*`) and their extras are mutually exclusive.
    pip_extra=(
        "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@61e3523d68184f5610bf045def9f93ba79ab3f4b",
    ),
    prefetch_weights=prefetch_weights,
)
