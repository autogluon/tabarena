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
# TabArena-Full run `tabldmplus_13102026` (816 tasks over 51 datasets), one direct fit per task on
# one RTX PRO 6000. `compute` is set by hand: the raw results record `num_gpus=0`, so the inferred
# value would be `cpu`.
tabldm_plus_method_metadata = MethodMetadata.system(
    method="Xiaomi-TabLDM+",
    name="Xiaomi-TabLDM+",
    suite="tabarena-2026-10-13",
    compute="gpu",
    date="2026-10-13",
    date_introduced="2026-10",
    reference_url="https://huggingface.co/occams/Xiaomi-TabLDM",
    organization="Xiaomi",
    commercial_use=True,
    license="Apache-2.0",
    tags=(),
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)


tabldm_plus_info = SystemInfo(
    system_cls=TabLDMPlusSystemModel,
    config_generator=gen_tabldm_plus,
    method_metadata=tabldm_plus_method_metadata,
    # TabLDM is not published on PyPI; pinned to a commit so the benchmarked code is fixed.
    # Keep in sync with the `tabldm_plus` extra in pyproject.toml and with the Xiaomi-TabLDM model
    # entry's pin, which runs the same commit.
    pip_extra=(
        "Xiaomi-TabLDM @ git+https://github.com/xiaomi-research/xiaomi-tabldm.git@d3fc3b7a0644631c8e3c18fa50629440e334f3f4",
    ),
    prefetch_weights=prefetch_weights,
)
