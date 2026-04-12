"""Run TabArena for feature selection benchmark with downstream model performance evaluation."""

from __future__ import annotations

from fr_cluster_setup import ALL_TASK_METADATA, FSBenchmarkConfig, TabArenaBenchmarkSetup

preprocessing_pipelines = FSBenchmarkConfig().get_default_preprocessing_configs(
    fs_methods=[
        "AccuracyFeatureSelector",
        "RandomFeatureSelector",
        "ANOVAFeatureSelector",
        "CFSFeatureSelector",
        "Chi2FeatureSelector",
        "DISRFeatureSelector",
        "GainRatioFeatureSelector",
        "GiniFeatureSelector",
        "ImpurityFeatureSelector",
        "InformationGainFeatureSelector",
        "INTERACTFeatureSelector",
        "MarkovBlanketFeatureSelector",
        "MIFeatureSelector",
        "mRMRFeatureSelector",
        "PearsonCorrelationFeatureSelector",
        "ReliefFFeatureSelector",
        "RFImportanceFeatureSelector",
        "SequentialBackwardEliminationFeatureSelector",
        "SequentialForwardSelectionFeatureSelector",
        "SymmetricalUncertaintyFeatureSelector",
        "LassoFeatureSelector",
        "ConsistencyFeatureSelector",
        "JMIFeatureSelector",
        "ElasticNetFeatureSelector",
        "OneRFeatureSelector",              # major OOM errors (tries to allocate one major array)
        "CARTFeatureSelector",              # OOM problems as well
        # "CMIMFeatureSelector",            # problems with fallback of features
        # "LaplacianScoreFeatureSelector",  # OOM, Segmentation fault issues
        # "tTestFeatureSelector",           # Does not work for regression
    ]
)

# Setup for CPU Methods (need to create another one for GPU models)
TabArenaBenchmarkSetup(
    # You could filter this to run less tasks
    task_metadata=ALL_TASK_METADATA,
    # Only run first three folds for now
    split_indices_to_run=["r0f0"],
    # Run methods for 5 configs (1 default + 4 random) each for now
    n_random_configs=0,
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    preprocessing_pipelines=preprocessing_pipelines,
).setup_jobs()