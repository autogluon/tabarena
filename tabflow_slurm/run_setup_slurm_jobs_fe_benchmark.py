"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling."""

# FIXME: merge this branch with minor fixes and support to get this running into mainline!
# https://github.com/LennartPurucker/autogluon/tree/minor_changes_for_tabarena

from __future__ import annotations

from itertools import product

from tabflow_slurm.setup_slurm_base import BenchmarkSetup

# TODO: improve/optimize how we can pass configs to the benchmark setup!
#   For now: we create a string with all the information and then decode the string
#   afterwards in run_tabarena_experiment.py
#   Ideally, we can support this in a nice way as we do for configs or other logic.
fs_methods = [
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
time_limit_fe = [0.33]
for fs_method, max_feature_threshold, proxy_model, time_limit in product(
    fs_methods, max_feature_thresholds, proxy_model_config, time_limit_fe
):
    config_str = f"FSBench__{fs_method}__{max_feature_threshold}__{proxy_model}__{time_limit}"
    preprocessing_pipelines.append(config_str)

# Add default config
preprocessing_pipelines.append("default")

# -- Get some proxy dataset to run for
# TODO: switch data foundry and public bucket (but we have an TabArena bucket now which we might want to use)
from tabarena.nips2025_utils.fetch_metadata import (
    load_curated_task_metadata,
)

metadata = load_curated_task_metadata()
metadata = metadata[
    metadata["dataset_id"].isin(
        [
            46953,  # QSAR-TID-11 (regression)
            46933,  # hiva_agnostic (multiclass)
            46912,  # Bioresponse (binary)
        ]
    )
]


# -- Feature Selection Example Benchmark Setup
# TODO:
#   - Finalize proxy model and its eval
#   - Decide on timeout, for both fit and fs or just fs? -> now it does for both
#   - Ensure random seeding works as we want it / add random order to methods that do iterative search of features
#   - Ensure how/if GPU models work with this (long shot)
#   - Investigate caching of feature selection (long shot)
BenchmarkSetup(
    benchmark_name="feature_selection_benchmark_example_1803",
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    n_random_configs=1,
    tabarena_lite=True,
    custom_metadata=metadata,
    preprocessing_pipelines=preprocessing_pipelines,
    time_limit=60 * 60 * 2,
    time_limit_with_preprocessing=True,
).setup_jobs()
