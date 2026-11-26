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
_common_kwargs = dict(
    compute="gpu",
    date_introduced="2026-11",
    reference_url="https://docs.priorlabs.ai/capabilities/thinking-mode",
    organization="Prior Labs",
    license="Proprietary (hosted API; Prior Labs terms of service)",
    commercial_use=False,
    tags=("closed-source-api",),
    fits_at_predict=False,
    verified=True,
)
_tabarena_kwargs = dict(suite="tabarena-2026-11-17", date="2026-11-17", **_common_kwargs)

# The system as the registry knows it: `--system TabPFN-3.5-Thinking` in the audit and the run scripts.
tabpfn_3_5_thinking_method_metadata = MethodMetadata.system(
    method="TabPFN-3.5-Thinking",
    name="TabPFN-3.5-Thinking",
    **_tabarena_kwargs,
)

# One hosted entry per thinking effort, like the Chakra-Tab presets. `xhigh` holds the TabArena run's
# `TabPFN-3.5-Thinking_c1_default` results (`tabpfn35thinking_17112026`, 1 h per fit). `max` holds the BeyondArena
# run's `TabPFN-3.5-Thinking_c2` results (`beyondarena_tabpfn35thinking_core2k_19112026`, 8 h per fit, the
# BeyondArena time limit for systems).
_hosted_kwargs = dict(cache_type="r2", cache_kwargs={"bucket": "tabarena", "prefix": "cache"})
tabpfn_3_5_thinking_xhigh_metadata = MethodMetadata.system(
    method="TabPFN-3.5-Thinking_xhigh",
    name="TabPFN-3.5-Thinking (xhigh)",
    **_tabarena_kwargs,
    **_hosted_kwargs,
)
tabpfn_3_5_thinking_max_metadata = MethodMetadata.system(
    method="TabPFN-3.5-Thinking_max",
    name="TabPFN-3.5-Thinking (max)",
    suite="beyondarena-2026-11-19",
    date="2026-11-19",
    **_common_kwargs,
    **_hosted_kwargs,
)

tabpfn_3_5_thinking_info = SystemInfo(
    system_cls=TabPFN35ThinkingSystemModel,
    config_generator=gen_tabpfn_3_5_thinking,
    method_metadata=tabpfn_3_5_thinking_method_metadata,
    pip_extra=("tabpfn-client==0.10.1",),
    prefetch_weights=None,
)
