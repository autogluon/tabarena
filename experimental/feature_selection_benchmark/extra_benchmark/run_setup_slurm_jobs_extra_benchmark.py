from pathlib import Path

import pandas as pd
import submitit

from experimental.feature_selection_benchmark.extra_benchmark.feature_selection_benchmark_runner import (
    ExtraBenchmarkJob,
    run_extra_benchmark_job,
)
from experimental.feature_selection_benchmark.tabarena_setup.fr_cluster_setup import (
    ALL_TASK_METADATA,
    FS_TIME_LIMIT,
    FSBenchmarkConfig,
)


def build_jobs():
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
            "LassoFeatureSelector",
            "LaplacianScoreFeatureSelector",
            "ConsistencyFeatureSelector",
            "JMIFeatureSelector",
            "OneRFeatureSelector",
            "ElasticNetFeatureSelector",
            "CMIMFeatureSelector",
            "CARTFeatureSelector",
        ]
    )
    task_ids = pd.read_csv(ALL_TASK_METADATA)["task_id_str"].tolist()

    modes = ["validity", "stability"]
    noises = [0.5, 0.75, 1.0]
    noise_types = ["gaussian"]

    jobs = []
    for mode in modes:
        for method_name in method_names:
            for task_id in task_ids:
                if mode == "validity":
                    for noise in noises:
                        for noise_type in noise_types:
                            jobs.append(
                                ExtraBenchmarkJob(
                                    mode=mode,
                                    method_name=method_name,
                                    task_id=task_id,
                                    noise=noise,
                                    noise_type=noise_type,
                                )
                            )
                else:
                    jobs.append(
                        ExtraBenchmarkJob(
                            mode=mode,
                            method_name=method_name,
                            task_id=task_id,
                        )
                    )
    return jobs


if __name__ == "__main__":
    jobs = build_jobs()

    executor = submitit.AutoExecutor(folder=Path("slurm_logs"))
    executor.update_parameters(
        timeout_min=FS_TIME_LIMIT,
        slurm_partition="default",
        cpus_per_task=8,
        mem_gb=32,
    )

    submitted_jobs = executor.map_array(run_extra_benchmark_job, jobs)
    print(f"Submitted {len(submitted_jobs)} jobs.")
