"""Code to use pre-defined benchmark environment and setup slurm jobs for (parallel) scheduling.

Special setup:
    # FIXME: merge this branch with minor fixes and support to get this running into mainline!
    (1) Install AutoGluon from https://github.com/LennartPurucker/autogluon/tree/minor_changes_for_tabarena

Usage:
    (1) download data foundry artifacts and run download_feature_selection_datasets.py

"""

from __future__ import annotations

from tabflow_slurm.setup_slurm_base_v2 import BenchmarkSetup2026
from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    get_metadata_for_benchmark_suite,
)
from tabflow_slurm.benchmarking_setup.download_feature_selection_datasets import DEFAULT_DATA_FOUNDRY_CACHE
from tabarena.benchmark.feature_selection_methods.feature_selection_benchmark_utils import (
    get_fs_benchmark_preprocessing_pipelines,
)


# TODO: improve/optimize how we can pass configs to the benchmark setup!
#   For now: we create a string with all the information and then decode the string
#   afterwards in run_tabarena_experiment.py
#   Ideally, we can support this in a nice way as we do for configs or other logic.
preprocessing_pipelines = get_fs_benchmark_preprocessing_pipelines(
    fs_methods=[
        "RandomFeatureSelector",
        "PearsonCorrelationFeatureSelector",
        "AccuracyFeatureSelector",
    ],
    proxy_model_config=["lgbm"],
    time_limit=[3600],
    total_budget=5,
    include_default=True,
)

# -- Feature Selection Example Benchmark Setup
# TODO:
#   - Finalize proxy model and its eval -> are we happy with its holdout validation?
#   - Finalize budget (evals per dataset) (b in formula)
#   - Ensure random seeding works as we want it / add random order to methods that do iterative search of features
#       -> ensure each config for one split of one dataset gets the same features/seed to simulate tuning correctly
#   - Investigate caching of feature selection (long shot)
#   - Finalize number of folds for downstream model evaluation
#   - Talk about problem with defining max_features based on n_features
#       -> must be done inside the split after dropping useless features (constant, duplicates).
#       -> so the value might change per split and cannot be really passed to the model. Added workaround.
BenchmarkSetup2026(
    benchmark_name="feature_selection_benchmark_example_2903",
    models=[
        ("LightGBM", "all"),
        ("RandomForest", "all"),
        ("Linear", "all"),
    ],
    n_random_configs=1,
    split_indices_to_run=["r0f0", "r0r1", "r0f2", "r1f0"],
    task_metadata=get_metadata_for_benchmark_suite(
        "feature_selection_benchmark_examples",
        data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE
    ),
    preprocessing_pipelines=preprocessing_pipelines,
    time_limit=3600,
    time_limit_for_model_agnostic_preprocessing=3600,
    dynamic_tabarena_validation_protocol=False,
).setup_jobs()