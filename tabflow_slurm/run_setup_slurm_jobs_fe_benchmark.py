"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

# FIXME: merge this branch with minor fixes and support to get this running into mainline!
# https://github.com/LennartPurucker/autogluon/tree/minor_changes_for_tabarena

from __future__ import annotations

from tabflow_slurm.setup_slurm_base import BenchmarkSetup
from itertools import product

# TODO: improve/optimize how we can pass configs to the benchmark setup!
#   For now: we create a string with all the information and then decode the string
#   afterwards in run_tabarena_experiment.py
#   Ideally, we can support this in a nice way as we do for configs or other logic.
fs_methods = [
    # ==== No Feature Selection (do we want to run this?)
    "default",
    # ==== Feature Selection Methods
    "RandomFeatureSelector",
    "PearsonCorrelationFeatureSelector",
    # "ANOVAFeatureSelector",
    # "CARTFeatureSelector",
    # "CFSFeatureSelector",
    # "Chi2FeatureSelector",
    # "CMIMFeatureSelector",
    # "ConsistencyFeatureSelector",
    # "DISRFeatureSelector",
    # "ElasticNetFeatureSelector",
    # "GainRatioFeatureSelector",
    # "GiniFeatureSelector",
    # "ImpurityFeatureSelector",
    # "InformationGainFeatureSelector",
    # "INTERACTFeatureSelector",
    # "JMIFeatureSelector",
    # "LaplacianScoreFeatureSelector",
    # "LassoFeatureSelector",
    # "MIFeatureSelector",
    # "mRMRFeatureSelector",
    # "OneRFeatureSelector",
    # "ReliefFFeatureSelector",
    # "RFImportanceFeatureSelector",
    # "SymmetricalUncertaintyFeatureSelector",
    # --- Require ProxyModelConfig
    "AccuracyFeatureSelector",
    # "SequentialBackwardEliminationFeatureSelector",
    # "SequentialForwardSelectionFeatureSelector",
    # --- Broke
    # "tTestFeatureSelector", # Only works for binary classification?
]
# TODO: make this int / per dataset?
max_feature_thresholds = [
    0.25,
    0.50,
    0.75,
]
proxy_model_config = [
    "lgbm",
]
preprocessing_pipelines = []
for fs_method, max_feature_threshold, proxy_model in product(
    fs_methods, max_feature_thresholds, proxy_model_config
):
    config_str = f"FSBench__{fs_method}__{max_feature_threshold}__{proxy_model}"
    preprocessing_pipelines.append(config_str)

# -- Feature Selection Example Benchmark Setup
# TODO:
#   - Ensure output has features etc (not yet)
#   - Ensure naming of model/job in output is clearly identifiable (not yet)
#   - Ensure we can pass time limit to FS methods and it works with the time limit of AG?
#   - Ensure how/if GPU models work with this
#   - Finalize proxy model and its eval
#   - Investigate caching of feature selection (long shot)
#   - Timeout for both fit and fs or just fs? -> now it does for both
BenchmarkSetup(
    benchmark_name="feature_selection_benchmark_example_1803",
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    n_random_configs=5,
    tabarena_lite=True,
    preprocessing_pipelines=preprocessing_pipelines,
    time_limit=60*60*2
).setup_jobs()