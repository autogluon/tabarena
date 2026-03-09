"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

from __future__ import annotations

from setup_slurm_base import BenchmarkSetup


# -- Benchmark TabPFN-v2.5 with Search Space 14/11/2025
BenchmarkSetup(
    benchmark_name="tabpfnv25_hpo_14112025",
    models=[
        ("RealTabPFN-v2.5", "all"),
    ],
    fs_methods=["Boruta"],
    num_gpus=1,
    configs_per_job=10,
    custom_model_constraints={
        "REALTABPFN-V2.5": {
            "max_n_samples_train_per_fold": 50_000,
            "max_n_features": 2000,
            "max_n_classes": 10,
        }
    },
).setup_jobs()

BenchmarkSetup(
    benchmark_name="tabpfnv25_fs_boruta_28112025",
    models=[
        # Only 25 configs due to large GPU memory requirements and long runtimes.
        # Plus not having enough large GPUs :(
        ("RealTabPFN-v2.5", 25),
    ],
    fs_methods=["Boruta"],
    num_gpus=1,
    configs_per_job=1,
    custom_model_constraints={
        "REALTABPFN-V2.5": {
            "max_n_samples_train_per_fold": 100_000,
            "max_n_features": 2000,
            "max_n_classes": 10,
            # To only run with big GPUs on big data.
            "min_n_samples_train_per_fold": 50_001,
        }
    },
    # Use H200 for large data jobs, similar to LimiX
    slurm_gpu_partition="alldlc2_gpu-h200",
    # H200 memory limit to override CPU estimates from AutoGluon
    fake_memory_for_estimates=140,
    # Ensure job scripts don't crash with above runs
    parallel_benchmark_fix="_large_vram",
).setup_jobs()
