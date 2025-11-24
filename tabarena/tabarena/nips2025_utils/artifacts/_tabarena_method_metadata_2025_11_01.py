from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-11-01",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    method_type="baseline",
    name_suffix=None,
    date="2025-11-01",
    verified=True,
    reference_url="https://arxiv.org/abs/2003.06505",
)

_gpu_kwargs = dict(
    compute="gpu",
    **_common_kwargs,
)

_cpu_kwargs = dict(
    compute="cpu",
    **_common_kwargs,
)

ag_140_eq_4h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_eq_4h8c",
    name="AutoGluon 1.4 (extreme, 4h)",
    **_gpu_kwargs,
)

ag_140_eq_1h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_eq_1h8c",
    name="AutoGluon 1.4 (extreme, 1h)",
    **_gpu_kwargs,
)

ag_140_eq_5m8c_metadata = MethodMetadata(
    method="AutoGluon_v140_eq_5m8c",
    name="AutoGluon 1.4 (extreme, 5m)",
    **_gpu_kwargs,
)

ag_140_bq_4h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_bq_4h8c",
    name="AutoGluon 1.4 (best, 4h)",
    **_cpu_kwargs,
)

ag_140_bq_1h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_bq_1h8c",
    name="AutoGluon 1.4 (best, 1h)",
    **_cpu_kwargs,
)

ag_140_bq_5m8c_metadata = MethodMetadata(
    method="AutoGluon_v140_bq_5m8c",
    name="AutoGluon 1.4 (best, 5m)",
    **_cpu_kwargs,
)

ag_140_hq_4h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_hq_4h8c",
    name="AutoGluon 1.4 (high, 4h)",
    **_cpu_kwargs,
)

ag_140_hq_5m8c_metadata = MethodMetadata(
    method="AutoGluon_v140_hq_5m8c",
    name="AutoGluon 1.4 (high, 5m)",
    **_cpu_kwargs,
)

ag_140_hqil_4h8c_metadata = MethodMetadata(
    method="AutoGluon_v140_hqil_4h8c",
    name="AutoGluon 1.4 (fast, 4h)",
    **_cpu_kwargs,
)

ag_140_hqil_5m8c_metadata = MethodMetadata(
    method="AutoGluon_v140_hqil_5m8c",
    name="AutoGluon 1.4 (fast, 5m)",
    **_cpu_kwargs,
)

methods_2025_11_01_ag: list[MethodMetadata] = [
    ag_140_eq_4h8c_metadata,
    ag_140_eq_1h8c_metadata,
    ag_140_eq_5m8c_metadata,
    ag_140_bq_4h8c_metadata,
    ag_140_bq_1h8c_metadata,
    ag_140_bq_5m8c_metadata,
    ag_140_hq_4h8c_metadata,
    ag_140_hq_5m8c_metadata,
    ag_140_hqil_4h8c_metadata,
    ag_140_hqil_5m8c_metadata,
]
