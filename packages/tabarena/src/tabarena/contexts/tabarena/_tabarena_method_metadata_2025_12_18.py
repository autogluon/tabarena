from __future__ import annotations

from tabarena.models._method_metadata import MethodMetadata

_common_kwargs = dict(
    suite="tabarena-2025-12-18",
    method_type="baseline",
    date="2025-12-18",
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

ag_150_eq_4h8c_metadata = MethodMetadata.tabarena_legacy_s3(
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
