"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# -- TabICLv2 14/02/2026
BenchmarkSetup(
    benchmark_name="tabpicl_v2_14022026",
    models=[
        ("TabICLv2", 0),
    ],
    num_gpus=1,
    configs_per_job=1,
    slurm_gpu_partition="alldlc2_gpu-h200",
    tabarena_lite=True,
).setup_jobs()

# # -- AutoGluon New Presets Benchmark 19/12/2025
# BenchmarkSetup(
#     benchmark_name="ag_experiment_191225",
#     models=[
#         (
#             "AutoGluon_extreme_v150_4h",
#             dict(
#                 fit_kwargs=dict(
#                     presets="https://ag-presets.s3.us-west-2.amazonaws.com/presets/extreme_v150.yaml",
#                 ),
#             ),
#         ),
#         (
#             "AutoGluon_extreme_noncommercial_v150_4h",
#             dict(
#                 fit_kwargs=dict(
#                     presets="https://ag-presets.s3.us-west-2.amazonaws.com/presets/extreme_noncommercial_v150.yaml",
#                 ),
#             ),
#         ),
#     ],
#     num_gpus=1,
#     time_limit=14400,
#     configs_per_job=1,
#     tabarena_lite=True,
# ).setup_jobs()

# # -- ConTexTab Benchmark / SAP RPT OSS 24/11/2025
# BenchmarkSetup(
#     benchmark_name="sap_rpt_oss_new_2411",
#     models=[
#         ("SAP-RPT-OSS", 0),
#     ],
#     num_gpus=1,
#     configs_per_job=1,
#     # Used only for a dataset with 230 features and 33k training samples.
#     # # Use H200 for large data jobs, similar to LimiX
#     # slurm_gpu_partition="alldlc2_gpu-h200",
#     # # H200 memory limit to override CPU estimates from AutoGluon
#     # fake_memory_for_estimates=140,
#     # model_agnostic_preprocessing=False,
#     # time_limit=5 * 60 * 60,
# ).setup_jobs()

# # -- Benchmark TabPFN-v2.5 with Search Space 14/11/2025
# BenchmarkSetup(
#     benchmark_name="tabpfnv25_hpo_14112025",
#     models=[
#         ("RealTabPFN-v2.5", "all"),
#     ],
#     num_gpus=1,
#     configs_per_job=10,
#     custom_model_constraints={
#         "REALTABPFN-V2.5": {
#             "max_n_samples_train_per_fold": 50_000,
#             "max_n_features": 2000,
#             "max_n_classes": 10,
#         }
#     },
# ).setup_jobs()
#
# BenchmarkSetup(
#     benchmark_name="tabpfnv25_hpo_14112025",
#     models=[
#         # Only 25 configs due to large GPU memory requirements and long runtimes.
#         # Plus not having enough large GPUs :(
#         ("RealTabPFN-v2.5", 25),
#     ],
#     num_gpus=1,
#     configs_per_job=1,
#     custom_model_constraints={
#         "REALTABPFN-V2.5": {
#             "max_n_samples_train_per_fold": 100_000,
#             "max_n_features": 2000,
#             "max_n_classes": 10,
#             # To only run with big GPUs on big data.
#             "min_n_samples_train_per_fold": 50_001,
#         }
#     },
#     # Use H200 for large data jobs, similar to LimiX
#     slurm_gpu_partition="alldlc2_gpu-h200",
#     # H200 memory limit to override CPU estimates from AutoGluon
#     fake_memory_for_estimates=140,
#     # Ensure job scripts don't crash with above runs
#     parallel_benchmark_fix="_large_vram",
# ).setup_jobs()
