from pathlib import Path

import pandas as pd
import submitit
from experimental.feature_selection_benchmark.tabarena_setup.fr_cluster_setup import (
    ALL_TASK_METADATA,
    FS_TIME_LIMIT,
    FSBenchmarkConfig,
)


def run_extra_pipeline(mode, method_name, task_id, noise, noise_type):
    """Function to run the extra pipeline."""
    print(
        f"Running extra pipeline with mode={mode}, method_name={method_name}, task_id={task_id}, noise={noise}, noise_type={noise_type}")
    # Add your pipeline logic here


if __name__ == "__main__":
    method_names = FSBenchmarkConfig().get_default_preprocessing_configs(
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
            "LassoFeatureSelector",  # just for regression but with label encoder for classification?
            "LaplacianScoreFeatureSelector",  # OOM, Segmentation fault issues
            "ConsistencyFeatureSelector",
            # selected_indices = np.where(S)[0].tolist(), UnboundLocalError: cannot access local variable 'S' where it is not associated with a value
            "JMIFeatureSelector",
            # time limit computed incorrectly, and error at remaining.remove(best_idx), ValueError: list.remove(x): x not in list
            "OneRFeatureSelector",
            # major OOM errors (tries to allocate one major array), wrong time limit computation,  max(accuracies, key=accuracies.get) -> max() iterable argument is empty
            "ElasticNetFeatureSelector",  # Only for classification
            "CMIMFeatureSelector",  # problems with time limit and fallback of features
            # "tTestFeatureSelector", # Does not work for regression
            "CARTFeatureSelector",  # Only implemented for classification, OOM problems as well
        ]
    )
    task_ids = pd.read_csv(ALL_TASK_METADATA)["task_id_str"].tolist()
    # Define the parameter grid
    modes = ["validity", "stability"]
    noises = [0.5, 0.75, 1.0]
    noise_types = ["gaussian"]

    # Create a SLURM executor
    executor = submitit.AutoExecutor(folder=Path("slurm_logs"))
    executor.update_parameters(
        timeout_min=FS_TIME_LIMIT,  # Job timeout in minutes
        slurm_partition="default",  # SLURM partition
        cpus_per_task=8,  # Number of CPUs per task
        mem_gb=32,  # Memory in GB
    )

    # Submit jobs
    with executor.batch():
        for mode in modes:
            for method_name in method_names:
                for task_id in task_ids:
                    if mode == "validity":
                        for noise in noises:
                            for noise_type in noise_types:
                                executor.submit(run_extra_pipeline, mode, method_name, task_id, noise, noise_type)
                    else:
                        executor.submit(run_extra_pipeline, mode, method_name, task_id, noise, noise_type)
