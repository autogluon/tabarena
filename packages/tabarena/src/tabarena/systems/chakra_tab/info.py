from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.chakra_tab.hpo import gen_chakra_tab
from tabarena.systems.chakra_tab.system import ChakraTabSystemModel

# Chakra-Tab is served through YHat Labs' API: the benchmark calls the endpoint, so it runs behind
# a remote API that cannot be inspected (tag closed-source-api). Fits ran on one A100 80 GB on the
# provider's side; `compute` records that.
chakra_tab_method_metadata = MethodMetadata.system(
    method="Chakra-Tab",
    name="Chakra-Tab",
    suite="tabarena-2026-10-04",
    compute="gpu",
    date="2026-10-04",
    date_introduced="2026-10",
    reference_url="https://yhatlabs.com",
    license="Proprietary (hosted API; YHat Labs terms of service)",
    commercial_use=False,
    tags=("closed-source-api",),
    verified=False,
)


chakra_tab_info = SystemInfo(
    system_cls=ChakraTabSystemModel,
    config_generator=gen_chakra_tab,
    method_metadata=chakra_tab_method_metadata,
    pip_extra=("requests>=2.31", "pyarrow>=14"),
    prefetch_weights=None,
)
