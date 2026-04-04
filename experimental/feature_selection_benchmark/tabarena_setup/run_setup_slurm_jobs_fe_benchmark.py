"""Run TabArena for feature selection benchmark with downstream model performance evaluation."""

from __future__ import annotations

from fr_cluster_setup import ALL_TASK_METADATA, FSBenchmarkConfig, TabArenaBenchmarkSetup

preprocessing_pipelines = FSBenchmarkConfig().get_default_preprocessing_configs(
    fs_methods=[
        "SequentialBackwardEliminationFeatureSelector",
        "RandomFeatureSelector",
        "PearsonCorrelationFeatureSelector",
    ]
)

# Setup for CPU Methods (need to create another one for GPU models)
TabArenaBenchmarkSetup(
    # You could filter this to run less tasks
    task_metadata=ALL_TASK_METADATA,
    # Only run first three folds for now
    split_indices_to_run=["r0f0", "r0r1", "r0f2"],
    # Run methods for 5 configs (1 default + 4 random) each for now
    n_random_configs=4,
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    preprocessing_pipelines=preprocessing_pipelines,
).setup_jobs()