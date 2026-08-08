"""Residual 2025-09-03 entries that don't fit the per-model `info.py` shape.

`tabflex_metadata` / `betatabpfn_metadata` are benchmark-result metadata for
models with no tabarena-side wrapper class (results were produced by external
code), and `TabPFNv2_GPU` similarly has no wrapper here. They're kept as
standalone `MethodMetadata` instances rather than `ModelInfo`. The suite's
AutoGluon entry lives in `tabarena.systems.autogluon.info` and is re-exported
here for the per-suite lists.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems.autogluon.info import ag_140_metadata

__all__ = ["ag_140_metadata", "betatabpfn_metadata", "tabflex_metadata"]

_common_kwargs = dict(
    suite="tabarena-2025-09-03",
)

# New methods (tabarena-2025-09-03)
tabflex_metadata = MethodMetadata.tabarena_legacy_s3(
    method="TabFlex_GPU",
    method_type="config",
    display_name="TabFlex",
    compute="gpu",
    date="2025-09-03",
    date_introduced="2025-06",
    ag_key="TABFLEX",
    model_key="TABFLEX_GPU",
    config_default="TabFlex_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=False,
    verified=False,
    reference_url="https://arxiv.org/abs/2506.05584",
    **_common_kwargs,
)
betatabpfn_metadata = MethodMetadata.tabarena_legacy_s3(
    method="BetaTabPFN_GPU",
    method_type="config",
    display_name="BetaTabPFN",
    compute="gpu",
    date="2025-09-03",
    date_introduced="2025-02",
    ag_key="BETA",
    model_key="BETA_GPU",
    config_default="BetaTabPFN_GPU_c1_BAG_L1",
    can_hpo=False,
    is_bag=True,
    verified=False,
    reference_url="https://arxiv.org/abs/2502.02527",
    **_common_kwargs,
)
