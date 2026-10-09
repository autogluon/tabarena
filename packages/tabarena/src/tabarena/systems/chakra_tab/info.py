from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.chakra_tab.hpo import gen_chakra_tab
from tabarena.systems.chakra_tab.system import ChakraTabSystemModel

# Chakra-Tab is served through YHat Labs' API: the benchmark calls the endpoint, so it runs behind
# a remote API that cannot be inspected (tag closed-source-api). Fits ran on one A100 80 GB on the
# provider's side; `compute` records that. The run is pinned to API version chakra-tab-2026-10, recorded
# with every result. The API fits and predicts in one call, so a result's `time_infer_s` holds the fit.
_common_kwargs = dict(
    suite="tabarena-2026-10-09",
    compute="gpu",
    date="2026-10-09",
    date_introduced="2026-10",
    reference_url="https://yhatlabs.com",
    license="Proprietary (hosted API; YHat Labs terms of service)",
    commercial_use=False,
    tags=("closed-source-api",),
    verified=False,  # set on upload, once the submitter has signed off on the run
)

# The system as the registry knows it: `--system Chakra-Tab` in the audit and the run scripts.
chakra_tab_method_metadata = MethodMetadata.system(method="Chakra-Tab", name="Chakra-Tab", **_common_kwargs)

# One hosted entry per preset, like the AutoGluon presets: the run's `Chakra-Tab_c1_default` results are
# `medium` (3-fold bagging), its `Chakra-Tab_c2_default` results are `full` (8-fold bagging).
chakra_tab_medium_metadata = MethodMetadata.system(
    method="Chakra-Tab_medium",
    name="Chakra-Tab (medium)",
    **_common_kwargs,
)
chakra_tab_full_metadata = MethodMetadata.system(
    method="Chakra-Tab_full",
    name="Chakra-Tab (full)",
    **_common_kwargs,
)

chakra_tab_info = SystemInfo(
    system_cls=ChakraTabSystemModel,
    config_generator=gen_chakra_tab,
    method_metadata=chakra_tab_method_metadata,
    pip_extra=("requests>=2.31", "pyarrow>=14"),
    prefetch_weights=None,
)
