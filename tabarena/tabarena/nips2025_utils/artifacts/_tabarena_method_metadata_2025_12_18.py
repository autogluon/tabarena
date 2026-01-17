from __future__ import annotations

from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata

_common_kwargs = dict(
    artifact_name="tabarena-2025-12-18",
    s3_bucket="tabarena",
    s3_prefix="cache",
    upload_as_public=True,
    has_raw=True,
    has_processed=True,
    has_results=True,
    method_type="baseline",
    name_suffix=None,
    date="2025-12-18",
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

ag_150_eq_4h8c_metadata = MethodMetadata(
    method="AutoGluon_v150_eq_4h8c",
    name="AutoGluon 1.5 (extreme, 4h)",
    **_gpu_kwargs,
)

# TODO: Need to run
# ag_150_bq_4h8c_metadata = MethodMetadata(
#     method="AutoGluon_v150_bq_4h8c",
#     name="AutoGluon 1.5 (best, 4h)",
#     **_cpu_kwargs,
# )

methods_2025_12_18_ag: list[MethodMetadata] = [
    ag_150_eq_4h8c_metadata,
    # ag_150_bq_4h8c_metadata,
]
