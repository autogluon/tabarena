"""Every benchmarked AutoGluon run, plus the wrapper that produces new ones.

The canonical home for AutoGluon's `MethodMetadata`. The dated
`contexts/tabarena/_tabarena_method_metadata_*.py` modules import from here and keep only
their per-suite lists, so a preset's identity is declared once.

The historical entries below were produced by the AMLB-style cluster runs, not by
`AutoGluonSystemModel`; the wrapper is the go-forward path for new AutoGluon results.
"""

from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata
from tabarena.systems._system_info import SystemInfo
from tabarena.systems.autogluon.hpo import gen_autogluon
from tabarena.systems.autogluon.system import AutoGluonSystemModel

_AUTOGLUON_PAPER = "https://arxiv.org/abs/2003.06505"

# Shared across every entry: AutoGluon is a system, is open-source, runs locally, and has no
# LLM in the loop, so it carries no tags.
_system_kwargs = dict(
    method_class="system",
    reference_url=_AUTOGLUON_PAPER,
)

# -- tabarena-2025-06-12 -------------------------------------------------------------------
ag_130_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v130",
    name="AutoGluon 1.3 (best, 4h)",
    suite="tabarena-2025-06-12",
    date="2025-06-12",
    date_introduced="2023-11",  # AutoGluon classic "best" preset
    method_type="baseline",
    compute="cpu",
    **_system_kwargs,
)

# -- tabarena-2025-09-03 -------------------------------------------------------------------
ag_140_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140",
    suite="tabarena-2025-09-03",
    method_type="baseline",
    display_name="AutoGluon 1.4 (4h)",
    compute="gpu",
    date="2025-09-03",
    date_introduced="2025-07-29",
    **_system_kwargs,
)

# -- tabarena-2025-11-01: the AutoGluon 1.4 preset x budget grid ---------------------------
_common_kwargs_2025_11_01 = dict(
    suite="tabarena-2025-11-01",
    method_type="baseline",
    date="2025-11-01",
    **_system_kwargs,
)

_gpu_kwargs = dict(
    compute="gpu",
    date_introduced="2025-07-29",  # AutoGluon 1.4.0 GitHub release ("extreme" preset ships with 1.4)
    **_common_kwargs_2025_11_01,
)

_cpu_kwargs = dict(
    compute="cpu",
    date_introduced="2023-11",  # AutoGluon classic "best"/"high"/"fast" presets
    **_common_kwargs_2025_11_01,
)

ag_140_eq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_eq_4h8c",
    name="AutoGluon 1.4 (extreme, 4h)",
    **_gpu_kwargs,
)

ag_140_eq_1h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_eq_1h8c",
    name="AutoGluon 1.4 (extreme, 1h)",
    **_gpu_kwargs,
)

ag_140_eq_5m8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_eq_5m8c",
    name="AutoGluon 1.4 (extreme, 5m)",
    **_gpu_kwargs,
)

ag_140_bq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_bq_4h8c",
    name="AutoGluon 1.4 (best, 4h)",
    **_cpu_kwargs,
)

ag_140_bq_1h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_bq_1h8c",
    name="AutoGluon 1.4 (best, 1h)",
    **_cpu_kwargs,
)

ag_140_bq_5m8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_bq_5m8c",
    name="AutoGluon 1.4 (best, 5m)",
    **_cpu_kwargs,
)

ag_140_hq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_hq_4h8c",
    name="AutoGluon 1.4 (high, 4h)",
    **_cpu_kwargs,
)

ag_140_hq_5m8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_hq_5m8c",
    name="AutoGluon 1.4 (high, 5m)",
    **_cpu_kwargs,
)

ag_140_hqil_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_hqil_4h8c",
    name="AutoGluon 1.4 (fast, 4h)",
    **_cpu_kwargs,
)

ag_140_hqil_5m8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v140_hqil_5m8c",
    name="AutoGluon 1.4 (fast, 5m)",
    **_cpu_kwargs,
)

# -- tabarena-2025-12-18 -------------------------------------------------------------------
ag_150_eq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
    method="AutoGluon_v150_eq_4h8c",
    name="AutoGluon 1.5 (extreme, 4h)",
    suite="tabarena-2025-12-18",
    method_type="baseline",
    date="2025-12-18",
    compute="gpu",
    date_introduced="2025-12",  # AutoGluon 1.5 "extreme" preset
    **_system_kwargs,
)

# TODO: Need to run
# ag_150_bq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
#     method="AutoGluon_v150_bq_4h8c",
#     name="AutoGluon 1.5 (best, 4h)",
#     suite="tabarena-2025-12-18",
#     method_type="baseline",
#     date="2025-12-18",
#     compute="cpu",
#     **_system_kwargs,
# )

# -- tabarena-2026-08-05: the released AutoGluon 1.6 presets -------------------------------
# TabArena-Full runs (816 tasks over 51 datasets) of the shipped presets, with only the fit
# `time_limit` raised to 4 hours. `compute` is set by hand: the runs used one GPU per task, but
# the raw results record `num_gpus=0`, so the inferred value would be `cpu`.
_common_kwargs_2026_08_05 = dict(
    suite="tabarena-2026-08-05",
    date="2026-08-05",
    compute="gpu",
    date_introduced="2026-08",  # AutoGluon 1.6 release
    reference_url=_AUTOGLUON_PAPER,
    verified=True,
    cache_type="r2",
    cache_kwargs={"bucket": "tabarena", "prefix": "cache"},
)

ag_160_eq_4h_metadata = MethodMetadata.system(
    method="AutoGluon_16_extreme",
    name="AutoGluon 1.6 (extreme, 4h)",
    **_common_kwargs_2026_08_05,
)

# The `noncommercial` preset is `extreme` plus TabPFN-3, whose license is not commercially
# permissive. `MethodMetadata` carries no license field, so the display name says it.
ag_160_noncomm_4h_metadata = MethodMetadata.system(
    method="AutoGluon_16_noncommercial",
    name="AutoGluon 1.6 (noncommercial, 4h)",
    **_common_kwargs_2026_08_05,
)


# Points at the latest registered AutoGluon run, matching how a model's `ModelInfo` points at
# its latest `MethodMetadata`.
autogluon_info = SystemInfo(
    system_cls=AutoGluonSystemModel,
    config_generator=gen_autogluon,
    method_metadata=ag_160_eq_4h_metadata,
)
