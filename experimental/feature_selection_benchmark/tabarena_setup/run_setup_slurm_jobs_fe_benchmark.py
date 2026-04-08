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
        # "LaplacianScoreFeatureSelector", # OOM, Segmentation fault issues
        # "ConsistencyFeatureSelector", # selected_indices = np.where(S)[0].tolist(), UnboundLocalError: cannot access local variable 'S' where it is not associated with a value
        # "JMIFeatureSelector", # time limit computed incorrectly, and error at remaining.remove(best_idx), ValueError: list.remove(x): x not in list
        # "OneRFeatureSelector", # major OOM errors (tries to allocate one major array), wrong time limit computation, max(accuracies, key=accuracies.get) -> max() iterable argument is empty
        "ElasticNetFeatureSelector",
        # "CMIMFeatureSelector", # problems with time limit and fallback of features
        # "tTestFeatureSelector", # Does not work for regression
        "CARTFeatureSelector", # Only implemented for classification, OOM problems as well
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