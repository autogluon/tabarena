from __future__ import annotations

from tabflow.cli.build_docker import build_docker
from tabflow.cli.launch_jobs import JobManager
from tabarena.models.lightgbm.generate import gen_lightgbm
from tabarena.nips2025_utils.tabarena_context import TabArenaContext
from tabarena.benchmark.experiment.experiment_constructor import Experiment, YamlExperimentSerializer

"""
# 1. Build the docker (ensure you use your own docker name to avoid overwriting other user's docker containers
bash ./tabarena/tabflow/docker/build_docker.sh tabarena tabarena-neerick 763104351884 097403188315 us-west-2
"""

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

"""
# 4. Run this script (working directory should be one level above the root of `tabarena`)

# 5. View the jobs and check their logs
https://us-west-2.console.aws.amazon.com/sagemaker/home?region=us-west-2#/jobs

# 6. View the result artifacts
https://us-west-2.console.aws.amazon.com/s3/buckets/{s3_bucket}?prefix={experiment_name}/
e.g: https://us-west-2.console.aws.amazon.com/s3/buckets/prateek-ag?prefix=tabarena-lightgbm-demo/

# 7. Download the artifacts to local
aws s3 cp --recursive "s3://prateek-ag/tabarena-lightgbm-demo" ../data/tabarena-lightgbm-demo/ --exclude "*.log"

# 8. Aggregate the local artifacts and evaluate them
Refer to `run_evaluate_lightgbm_demo.py`
"""

# Edit to use your own account
REPO_NAME = "tabarena"
TAG = "tabarena-neerick"
SOURCE_ACCOUNT = "763104351884"
TARGET_ACCOUNT = "097403188315"
REGION = "us-west-2"


# TODO: Add example for custom model / non-AG model
if __name__ == "__main__":
    build_docker_flag = True
    if build_docker_flag:
        build_docker(
            repo_name=REPO_NAME,
            tag=TAG,
            source_account=SOURCE_ACCOUNT,
            target_account=TARGET_ACCOUNT,
            region=REGION,
        )

    docker_image_uri = f"{TARGET_ACCOUNT}.dkr.ecr.{REGION}.amazonaws.com/{REPO_NAME}:{TAG}"
    sagemaker_role = f"arn:aws:iam::{TARGET_ACCOUNT}:role/service-role/AmazonSageMaker-ExecutionRole-20250128T153145"

    tabarena_context = TabArenaContext()
    task_metadata = tabarena_context.task_metadata.copy()  # metadata about the available datasets

    experiment_name = "tabarena-lightgbm-demo"  # The experiment name, used as the s3 path prefix for the saved files.
    s3_bucket = "prateek-ag"  # The s3 bucket to save results to
    s3_dataset_cache = "s3://tabarena/openml/openml_cache"

    max_concurrent_jobs = 10  # the max number of instances running jobs at the same time (values 1 - 15000)
    batch_size = 1  # The number of jobs to give to a single instance to run sequentially.
    wait = True  # If True, will only return when all jobs are finished running.
    ignore_cache = False  # If True, will overwrite prior results in s3. If False, will skip runs for finished jobs.

    region_name = "us-west-2"  # AWS region to create instances. Keep as us-west-2.
    instance_type = "ml.m6i.2xlarge"  # options: ml.m6i.2xlarge (cpu, 15k cap), ml.g6.2xlarge (gpu, 400 cap)

    use_yaml_file = False  # two alternative ways of getting the methods info
    if use_yaml_file:
        methods_file = "./tabarena/tabflow/configs/configs_lightgbm_demo.yaml"  # 2 lightgbm configs
        methods: list[Experiment] = YamlExperimentSerializer.from_yaml(path=methods_file)
    else:
        methods: list[Experiment] = gen_lightgbm.generate_all_bag_experiments(num_random_configs=1)
    methods_content = YamlExperimentSerializer.to_yaml_str(methods)
    methods_as_dict: list[dict] = [m.to_yaml_dict() for m in methods]

    toy_run = True

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
