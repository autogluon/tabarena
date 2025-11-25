from __future__ import annotations

from tabflow.cli.launch_jobs import JobManager
from tabarena.models.lr.generate import gen_linear
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.benchmark.experiment.experiment_constructor import Experiment, YamlExperimentSerializer

"""
# 1. Build the docker (ensure you use your own docker name to avoid overwriting other user's docker containers
bash ./tabarena/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
"""
docker_image_uri = "097403188315.dkr.ecr.us-west-2.amazonaws.com/tabarena:tabarena-neerick"

"""
# 2. Need to set aws default region to us-west-2
aws configure
Default region name [None]: us-west-2
less ~/.aws/config
# delete config to revert
TODO: Make this not be required
"""

"""
# 3. Create a sagemaker role (only for fresh AWS account)
TODO
"""
sagemaker_role = "arn:aws:iam::097403188315:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"

"""
# 4. Run this script (working directory should be one level above the root of `tabarena`)

# 5. View the jobs and check their logs
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs

# 6. View the result artifacts
https://us-west-2.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={experiment_name}/

# 7. Download the artifacts to local
from tabarena.utils.s3_downloader import copy_s3_prefix_to_local
copy_s3_prefix_to_local(
    bucket=s3_bucket,
    prefix=experiment_name,
    dest_dir=f"../data/{experiment_name}/",
    max_workers=64,
    exclude=["*.log"],
)

# 8. Aggregate the local artifacts and evaluate them
Refer to `run_evaluate_linear_model.py`
"""


if __name__ == "__main__":
    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata.copy()  # metadata about the available datasets

    experiment_name = "tabarena-lr-2025-10-16"  # The experiment name, used as the s3 path prefix for the saved files.
    s3_bucket = "prateek-ag"  # The s3 bucket to save results to
    s3_dataset_cache = "s3://tabarena/openml/openml_cache"

    max_concurrent_jobs = 2000  # the max number of instances running jobs at the same time (values 1 - 15000)
    batch_size = 512  # The number of jobs to give to a single instance to run sequentially.
    wait = True  # If True, will only return when all jobs are finished running.
    ignore_cache = False  # If True, will overwrite prior results in s3. If False, will skip runs for finished jobs.

    region_name = "us-west-2"  # AWS region to create instances. Keep as us-west-2.
    instance_type = "ml.m6i.2xlarge"  # options: ml.m6i.2xlarge (cpu, 15k cap), ml.g6.2xlarge (gpu, 400 cap)

    methods: list[Experiment] = gen_linear.generate_all_bag_experiments(num_random_configs=200)
    methods_content = YamlExperimentSerializer.to_yaml_str(methods)
    methods_as_dict: list[dict] = [m.to_yaml_dict() for m in methods]

    toy_run = False

    # toy run
    if toy_run:
        # only run 3 small datasets
        # only run 1 fold and 1 repeat per dataset
        datasets = ["anneal", "credit-g", "diabetes"]
        task_metadata["n_folds"] = 1
        task_metadata["n_repeats"] = 1
    else:
        datasets = list(task_metadata["name"])

    job_manager = JobManager(
        experiment_name=experiment_name,
        task_metadata=task_metadata,
        methods_content=methods_content,
        max_concurrent_jobs=max_concurrent_jobs,
        s3_bucket=s3_bucket,
        wait=wait,
        instance_type=instance_type,
        batch_size=batch_size,
        sagemaker_role=sagemaker_role,
        docker_image_uri=docker_image_uri,
        s3_dataset_cache=s3_dataset_cache,
    )

    # get the list of jobs to execute
    tasks_batched = job_manager.get_tasks_batched(
        datasets=datasets,
        methods=methods_as_dict,
        batch_size=batch_size,
        ignore_cache=ignore_cache,
    )

    # run the jobs
    job_manager.run_tasks_batched(task_batch_lst=tasks_batched)
