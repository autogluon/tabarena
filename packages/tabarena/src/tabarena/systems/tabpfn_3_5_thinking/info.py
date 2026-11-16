from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.tabpfn_3_5_thinking.hpo import gen_tabpfn_3_5_thinking
from tabarena.systems.tabpfn_3_5_thinking.system import TabPFN35ThinkingSystemModel

# TabPFN-3.5-Thinking is served through Prior Labs' API: the benchmark calls it with the official client, so it runs
# behind a remote API that cannot be inspected (tag closed-source-api). It does not use LLMs, so `with-llm` does not
# apply. The API runs on GPUs; `compute` records that. The client fits and predicts in separate calls, so a result's
# `time_train_s` holds the fit (`fits_at_predict=False`). The client version, the service version and the server-side
# timings are recorded with every result.
tabpfn_3_5_thinking_method_metadata = MethodMetadata.system(
    method="TabPFN-3.5-Thinking",
    name="TabPFN-3.5-Thinking",
    suite="tabarena-2026-11-11",
    compute="gpu",
    date="2026-11-11",
    date_introduced="2026-09",
    reference_url="https://docs.priorlabs.ai/capabilities/thinking-mode",
    organization="Prior Labs",
    license="Proprietary (hosted API; Prior Labs terms of service)",
    commercial_use=False,
    tags=("closed-source-api",),
    fits_at_predict=False,
    verified=False,
)

tabpfn_3_5_thinking_info = SystemInfo(
    system_cls=TabPFN35ThinkingSystemModel,
    config_generator=gen_tabpfn_3_5_thinking,
    method_metadata=tabpfn_3_5_thinking_method_metadata,
    pip_extra=("tabpfn-client==0.8.0",),
    prefetch_weights=None,
)
