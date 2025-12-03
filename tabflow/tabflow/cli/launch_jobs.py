from __future__ import annotations

import boto3
import sagemaker
import argparse
import logging
import uuid
import pandas as pd

from autogluon.common.savers import save_pd
from botocore.config import Config
from datetime import datetime
from pathlib import Path
from tabflow.core.resource_manager import TrainingJobResourceManager
from tabflow.utils.utils import sanitize_job_name, yaml_to_methods, create_batch
from tabflow.utils.s3_utils import check_s3_file_exists, upload_tasks_json
from tabflow.utils.logging_utils import setup_logging
from tabflow.utils.constants import DOCKER_IMAGE_ALIASES

logger = setup_logging(level=logging.ERROR)


# TODO: Integrate this into JobManager
class Task:
    def __init__(self, dataset: str, repeat: int, fold: int, method: dict):
        self.dataset = dataset
        self.repeat = repeat
        self.fold = fold
        self.method = method

    @property
    def method_name(self) -> str:
        return self.method["name"]


class JobManager:
    def __init__(
        self,
        experiment_name: str,
        task_metadata: pd.DataFrame,
        s3_bucket: str,
        docker_image_uri: str,
        sagemaker_role: str,
        entry_point: str = "evaluate.py",
        source_dir: str = str(Path(__file__).parent),
        instance_type: str = "ml.m6i.2xlarge",
        keep_alive_period_in_seconds: int = 120,
        limit_runtime: int = 24 * 60 * 60,
        max_concurrent_jobs: int = 30,
        max_retry_attempts: int = 20,
        batch_size: int = 1,
        methods_content: str | None = None,
        methods_file: str | None = None,
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        add_timestamp: bool = False,
        wait: bool = True,
        stop_wait_on_fail: bool = False,
        s3_dataset_cache: str = None,
    ):
        """

        Parameters
        ----------
        experiment_name:
            Name of the experiment
        task_metadata : pd.DataFrame
            The task metadata. Must contain "tid" and "name" columns.
        entry_point:
            The Python script to run in sagemaker training job
        source_dir:
            Directory containing the training code (here the entry point)
        instance_type:
            SageMaker instance type
        docker_image_uri:
            Docker image to use URI or alias in constants.py
        sagemaker_role:
            AWS IAM role for SageMaker
        aws_profile:
            AWS profile name
        hyperparameters:
            Dictionary of hyperparameters to pass to the training script
        keep_alive_period_in_seconds:
            Idle time before terminating the instance
        limit_runtime:
            Maximum running time in seconds
        methods_file:
            Path to the YAML file containing methods
        methods_content:
            The YAML str content for methods. Alternative input to `methods_file` which avoids needing a YAML file.
        max_concurrent_jobs:
            Maximum number of concurrent jobs, based on account limit
        S3 bucket:
            S3 bucket to store the results
        add_timestamp:
            Whether to add a timestamp to the experiment name
        wait:
            Whether to wait for all jobs to complete (no-wait from CLI)
        stop_wait_on_fail:
            Whether to stop waiting if any of the jobs fail - only applied if wait=True
        batch_size:
            Number of models to batch for each task
        s3_dataset_cache:
            Full S3 URI for OpenML dataset cache (format: s3://bucket/prefix),
            note that after the prefix the following will be appended to the path -
            tasks/{task_id}/org/openml/www/tasks/{task_id}, where the xml and arff is expected to be situated

        """
        if add_timestamp:
            timestamp = datetime.now().strftime("%d-%b-%Y-%H:%M:%S.%f")[:-3]
            experiment_name = f"{experiment_name}-{timestamp}"
        if docker_image_uri in DOCKER_IMAGE_ALIASES:
            logger.info(
                f"Expanding docker_image_uri alias '{docker_image_uri}' -> '{DOCKER_IMAGE_ALIASES[docker_image_uri]}'"
            )
            docker_image_uri = DOCKER_IMAGE_ALIASES[docker_image_uri]

        self.task_metadata = self._process_task_metadata(task_metadata=task_metadata)
        # FIXME: Don't do this. Just rely on `name` or `tid`
        self.datasets_to_tids = self.task_metadata.set_index("dataset")["tid"].to_dict()

        self.experiment_name = experiment_name
        if methods_content is not None:
            assert methods_file is None, f"Only one of `methods_file`, `methods_content` can be specified."
        elif methods_file is None:
            raise AssertionError(f"Must specify one of `methods_file`, `methods_content`.")
        self.methods_content = methods_content
        self.methods_file = methods_file
        self.s3_bucket = s3_bucket
        self.docker_image_uri = docker_image_uri
        self.sagemaker_role = sagemaker_role
        self.entry_point = entry_point
        self.source_dir = source_dir
        self.instance_type = instance_type
        self.keep_alive_period_in_seconds = keep_alive_period_in_seconds
        self.limit_runtime = limit_runtime
        self.max_concurrent_jobs = max_concurrent_jobs
        self.max_retry_attempts = max_retry_attempts
        self.batch_size = batch_size
        self.aws_profile = aws_profile
        self.hyperparameters = hyperparameters
        self.add_timestamp = add_timestamp
        self.wait = wait
        self.stop_wait_on_fail = stop_wait_on_fail
        self.s3_dataset_cache = s3_dataset_cache
        self._job_count = 0  # Used to ensure unique job names

        # Create boto3 session
        self.boto_session = boto3.Session(profile_name=self.aws_profile) if self.aws_profile else boto3.Session()
        # Create SageMaker session + retry config
        self.retry_config = Config(
            connect_timeout=5,
            read_timeout=10,
            retries={'max_attempts': max_retry_attempts,
                     'mode': 'adaptive',
                     }
        )
        self.sagemaker_client = self.boto_session.client('sagemaker', config=self.retry_config)
        self.sagemaker_session = sagemaker.Session(boto_session=self.boto_session, sagemaker_client=self.sagemaker_client)
        # Create S3 client
        self.s3_client = self.boto_session.client('s3', config=self.retry_config)

        # Initialize the resource manager
        self.resource_manager = TrainingJobResourceManager(
            sagemaker_client=self.sagemaker_client,
            max_concurrent_jobs=self.max_concurrent_jobs,
        )

        self._is_methods_file_uploaded = False
        self._is_task_metadata_file_uploaded = False

    @property
    def methods_s3_key(self) -> str:
        return f"{self.experiment_name}/config/methods_config.yaml"

    @property
    def methods_s3_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.methods_s3_key}"

    @property
    def task_metadata_s3_path(self) -> str:
        return f"s3://{self.s3_bucket}/{self.experiment_name}/task_metadata.csv"

    def _process_task_metadata(self, task_metadata: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure task_metadata is in the proper format.

        Parameters
        ----------
        task_metadata: pd.DataFrame

        Returns
        -------

        """
        assert isinstance(task_metadata, pd.DataFrame)
        task_metadata = task_metadata.copy(deep=True)
        assert "tid" in task_metadata
        assert "name" in task_metadata
        assert len(task_metadata.drop_duplicates(subset=["name"])) == len(task_metadata)
        assert len(task_metadata.drop_duplicates(subset=["tid"])) == len(task_metadata)
        if "dataset" in task_metadata:
            is_matching_names = (task_metadata["name"] == task_metadata["dataset"]).all()
            if not is_matching_names:
                print(f"WARNING: NAME NOT MATCHING IN TASK_METADATA! THIS MAY LEAD TO MAJOR ISSUES. PLEASE AVOID THIS.")
        else:
            # FIXME: Don't do this. Just rely on `name` or `tid`
            task_metadata["dataset"] = task_metadata["name"]
        assert len(task_metadata.drop_duplicates(subset=["dataset"])) == len(task_metadata)
        return task_metadata

    def upload_methods_file_to_s3(self):
        self._is_methods_file_uploaded = True
        if self.methods_file is not None:
            self.s3_client.upload_file(self.methods_file, self.s3_bucket, self.methods_s3_key)
        else:
            assert self.methods_content is not None
            self.s3_client.put_object(
                Bucket=self.s3_bucket,
                Key=self.methods_s3_key,
                Body=self.methods_content.encode("utf-8"),
            )

    def upload_task_metadata_file_to_s3(self):
        self._is_task_metadata_file_uploaded = True
        save_pd.save(path=self.task_metadata_s3_path, df=self.task_metadata)

    @classmethod
    def load_methods_from_yaml(cls, methods_file: str):
        methods = yaml_to_methods(methods_file=methods_file)
        return methods

    def get_tasks_dense(
        self,
        datasets: list[str],
        repeats: list[int],
        folds: list[int],
        methods: list[dict],
    ) -> list[tuple[str, int, int, dict]]:
        for dataset in datasets:
            assert dataset in self.datasets_to_tids
        tasks = [(dataset, repeat, fold, method) for dataset in datasets for repeat in repeats for fold in folds for method in methods]
        return tasks

    def get_tasks_batched(
        self,
        datasets: list[str],
        methods: list[dict],
        batch_size: int | None = None,
        ignore_cache: bool = False,
    ) -> list[list[tuple[str, int, int, dict]]]:
        if batch_size is None:
            batch_size = self.batch_size
        for dataset in datasets:
            assert dataset in self.datasets_to_tids
        repeats_fold_dict = {}
        for dataset in datasets:
            cur_dataset_metadata = self.task_metadata[self.task_metadata["name"] == dataset].iloc[0]
            if "batch_size" in self.task_metadata.columns:
                cur_batch_size = cur_dataset_metadata["batch_size"]
                if cur_batch_size is None:
                    cur_batch_size = batch_size
            else:
                cur_batch_size = batch_size
            cur_repeats = cur_dataset_metadata["n_repeats"]
            cur_folds = cur_dataset_metadata["n_folds"]
            if (cur_repeats, cur_folds, cur_batch_size) not in repeats_fold_dict.keys():
                repeats_fold_dict[(cur_repeats, cur_folds, cur_batch_size)] = []
            repeats_fold_dict[(cur_repeats, cur_folds, cur_batch_size)].append(dataset)

        repeats_folds_tasks_dict = dict()
        for (cur_repeats, cur_folds, cur_batch_size), cur_datasets in repeats_fold_dict.items():
            repeats = [i for i in range(cur_repeats)]
            folds = [i for i in range(cur_folds)]
            cur_tasks_dense = self.get_tasks_dense(
                datasets=cur_datasets,
                repeats=repeats,
                folds=folds,
                methods=methods,
            )
            repeats_folds_tasks_dict[(cur_repeats, cur_folds, cur_batch_size)] = cur_tasks_dense

        tasks_batch_lst = []

        for (cur_repeats, cur_folds, cur_batch_size), cur_tasks_dense in repeats_folds_tasks_dict.items():
            if not ignore_cache:
                cur_tasks_dense = self.filter_to_only_uncached_tasks(tasks=cur_tasks_dense, verbose=True)
            cur_tasks_batch = self.batch_tasks(tasks=cur_tasks_dense, batch_size=cur_batch_size)
            tasks_batch_lst.append(cur_tasks_batch)
        tasks_batch_combined = [task_batch for tasks_batch in tasks_batch_lst for task_batch in tasks_batch]
        return tasks_batch_combined

    def run_tasks(self, tasks: list[tuple[str, int, int, dict]], check_cache: bool = False, batch_size: int | None = None):
        if batch_size is None:
            batch_size = self.batch_size
        if len(tasks) == 0:
            logger.info(f"All jobs are already completed.")
            return

        # TODO: Only include tasks without caches! Keep adding to task_batch until it is of size self.batch_size
        task_batch_lst = self.batch_tasks(tasks=tasks, batch_size=batch_size)
        logger.info(
            f"Preparing to launch jobs with batch size of {batch_size}"
        )

        self.run_tasks_batched(task_batch_lst=task_batch_lst, check_cache=check_cache)

    def run_tasks_batched(self, task_batch_lst: list[list[tuple[str, int, int, dict]]], check_cache: bool = False):
        total_jobs = len(task_batch_lst)
        self.resource_manager.total_jobs = total_jobs

        logger.info(
            f"Preparing to launch {total_jobs} jobs with max concurrency of {self.max_concurrent_jobs}"
        )
        logger.info(
            f"Instance keep-alive period set to "
            f"{self.keep_alive_period_in_seconds} seconds to enable warm-starts"
        )

        for task_batch in task_batch_lst:
            self.run_task_batch(tasks=task_batch, check_cache=check_cache)

        if self.wait:
            self.resource_manager.wait_for_all_jobs(
                s3_client=self.s3_client, s3_bucket=self.s3_bucket, stop_wait_on_fail=self.stop_wait_on_fail
            )

    def batch_tasks(
        self,
        tasks: list[tuple[str, int, int, dict]],
        batch_size: int,
    ) -> list[list[tuple[str, int, int, dict]]]:
        task_batch_lst = list(create_batch(tasks, batch_size))
        return task_batch_lst

    def check_if_task_is_cached(self, dataset: dict, repeat: int, fold: int, method, verbose: bool = False):
        method_name = method['name']
        cache_path = f"{self.experiment_name}/data/{method_name}/{self.datasets_to_tids[dataset]}/{repeat}_{fold}"
        cache_name = f"{cache_path}/results.pkl"
        if check_s3_file_exists(s3_client=self.s3_client, bucket=self.s3_bucket, cache_name=cache_name):
            if verbose:
                logger.info(f"Cache exists for {method_name} on dataset {dataset} repeat {repeat} fold {fold}. Skipping job launch.")
                logger.info(f"Cache path: s3://{self.s3_bucket}/{cache_path}\n")
            return True
        else:
            return False

    def _task_cache_path(self, task: tuple) -> str:
        dataset, repeat, fold, method = task
        method_name = method["name"]
        cache_path = f"{self.experiment_name}/data/{method_name}/{self.datasets_to_tids[dataset]}/{repeat}_{fold}"
        return cache_path

    def _task_cache_name(self, task: tuple) -> str:
        return f"{self._task_cache_path(task=task)}/results.pkl"

    def make_job_name(self, task: tuple, suffix: str = None):
        dataset, repeat, fold, method = task
        method_name = method["name"]

        # Create a unique job name
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        if suffix is None:
            unique_id = str(uuid.uuid4().int)[:8]
        else:
            unique_id = str(uuid.uuid4().int)[:4]
        base_name = f"{dataset[:4]}-r{repeat}-f{fold}-{method_name[:12]}"
        if suffix is not None:
            base_name = f"{base_name}-{suffix}"
        base_name = f"{base_name}-{timestamp}-{unique_id}"
        job_name = sanitize_job_name(base_name)
        return job_name

    def get_is_cached_lst(self, tasks: list[tuple], verbose: bool = False) -> list[bool]:
        num_tasks = len(tasks)
        num_cached = 0
        is_cached_lst = []
        for i, (dataset, repeat, fold, method) in enumerate(tasks):
            is_cached = self.check_if_task_is_cached(dataset=dataset, repeat=repeat, fold=fold, method=method, verbose=False)

            is_cached_lst.append(is_cached)

            if is_cached:
                num_cached += 1
            if verbose and (i % 100 == 0):
                print(f"{i + 1}/{num_tasks}\tCached: {num_cached}\t({dataset}, {repeat}, {fold}, {method['name']})")
        return is_cached_lst

    def filter_to_only_uncached_tasks(self, tasks: list[tuple], verbose: bool = False) -> list[tuple]:
        is_cached_lst = self.get_is_cached_lst(tasks=tasks, verbose=verbose)
        return [task for task, is_cached in zip(tasks, is_cached_lst) if not is_cached]

    def launch_estimator(self, job_hyperparameters):
        # Create the estimator
        estimator = sagemaker.estimator.Estimator(
            entry_point=self.entry_point,
            source_dir=self.source_dir,
            image_uri=self.docker_image_uri,
            role=self.sagemaker_role,
            instance_count=1,
            instance_type=self.instance_type,
            sagemaker_session=self.sagemaker_session,
            hyperparameters=job_hyperparameters,
            keep_alive_period_in_seconds=self.keep_alive_period_in_seconds,
            max_run=self.limit_runtime,
            disable_profiler=True,  # Prevent debug profiler from running
            # output_path=f"s3://{s3_bucket}/{experiment_name}/data/output"  #TBD: What artifact to save here?
        )
        return estimator

    def run_task_batch(self, tasks: list[tuple], check_cache: bool = False):
        if not self._is_methods_file_uploaded:
            self.upload_methods_file_to_s3()
        if not self._is_task_metadata_file_uploaded:
            self.upload_task_metadata_file_to_s3()

        if check_cache:
            tasks = self.filter_to_only_uncached_tasks(tasks=tasks, verbose=True)
        if not tasks:
            logger.info(f"All tasks in batch are cached. Skipping job launch.")
            return

        self.resource_manager.wait_for_available_slot(s3_client=self.s3_client, s3_bucket=self.s3_bucket)

        self._job_count += 1
        job_name_suffix = str(self._job_count)
        job_name = self.make_job_name(task=tasks[0], suffix=job_name_suffix)
        cache_path = self._task_cache_path(task=tasks[0])

        tasks_json = []
        for dataset, repeat, fold, method in tasks:
            tasks_json.append({
                "dataset": dataset,
                "fold": fold,  # NOTE: Can be a 'str' as well, refer to Estimators in SM docs
                "repeat": repeat,
                "method_name": method["name"],
            })

        # Unique s3 key for a task
        tasks_s3_key = f"{self.experiment_name}/config/tasks/{job_name}_tasks.json"
        tasks_s3_path = upload_tasks_json(self.s3_client, tasks_json, self.s3_bucket, tasks_s3_key)

        # Update hyperparameters for this job
        job_hyperparameters = self.hyperparameters.copy() if self.hyperparameters else {}
        job_hyperparameters.update({
            "experiment_name": self.experiment_name,
            "task_metadata_path": self.task_metadata_s3_path,
            "tasks_s3_path": tasks_s3_path,
            "s3_bucket": self.s3_bucket,
            "methods_s3_path": self.methods_s3_path,
        })
        if self.s3_dataset_cache is not None:
            job_hyperparameters["s3_dataset_cache"] = self.s3_dataset_cache

        estimator = self.launch_estimator(job_hyperparameters=job_hyperparameters)

        # Launch the training job
        estimator.fit(wait=False, job_name=job_name)
        self.resource_manager.add_job(job_name=job_name, cache_path=cache_path)


def launch_jobs(
        experiment_name: str,
        methods_file: str,
        s3_bucket: str,
        docker_image_uri: str,
        sagemaker_role: str,
        task_metadata: pd.DataFrame,
        entry_point: str = "evaluate.py",
        source_dir: str = str(Path(__file__).parent),
        instance_type: str = "ml.m6i.4xlarge",
        keep_alive_period_in_seconds: int = 120,
        limit_runtime: int = 24 * 60 * 60,
        max_concurrent_jobs: int = 30,
        max_retry_attempts: int = 20,
        batch_size: int = 1,
        aws_profile: str | None = None,
        hyperparameters: dict = None,
        datasets: list = None,
        folds: list = None,
        add_timestamp: bool = False,
        wait: bool = True,
        s3_dataset_cache: str = None,
) -> None:
    """
    Launch multiple SageMaker training jobs.

    Args:
        experiment_name: Name of the experiment
        context_name: Name of the TabArena context
        entry_point: The Python script to run in sagemaker training job
        source_dir: Directory containing the training code (here the entry point)
        instance_type: SageMaker instance type
        docker_image_uri: Docker image to use URI or alias in constants.py
        sagemaker_role: AWS IAM role for SageMaker
        aws_profile: AWS profile name
        hyperparameters: Dictionary of hyperparameters to pass to the training script
        keep_alive_period_in_seconds: Idle time before terminating the instance 
        limit_runtime: Maximum running time in seconds
        datasets: List of datasets to evaluate
        folds: List of folds to evaluate
        methods_file: Path to the YAML file containing methods
        max_concurrent_jobs: Maximum number of concurrent jobs, based on account limit
        S3 bucket: S3 bucket to store the results
        add_timestamp: Whether to add a timestamp to the experiment name
        wait: Whether to wait for all jobs to complete (no-wait from CLI)
        batch_size: Number of models to batch for each task
        s3_dataset_cache: Full S3 URI for OpenML dataset cache (format: s3://bucket/prefix), note that after prefix
        the following will be appended to the path - tasks/{task_id}/org/openml/www/tasks/{task_id}, where the xml and arff is expected to be situated
    """

    job_manager = JobManager(
        experiment_name=experiment_name,
        task_metadata=task_metadata,
        methods_file=methods_file,
        max_concurrent_jobs=max_concurrent_jobs,
        s3_bucket=s3_bucket,
        wait=wait,
        instance_type=instance_type,
        batch_size=batch_size,
        sagemaker_role=sagemaker_role,
        docker_image_uri=docker_image_uri,
        s3_dataset_cache=s3_dataset_cache,
    )

    methods = job_manager.load_methods_from_yaml(methods_file=job_manager.methods_file)

    dense_tasks = job_manager.get_tasks_dense(
        datasets=datasets,
        folds=folds,
        methods=methods,
    )

    uncached_tasks = job_manager.filter_to_only_uncached_tasks(tasks=dense_tasks, verbose=True)

    print(f"{len(uncached_tasks)}/{len(dense_tasks)} Uncached Tasks")

    job_manager.run_tasks(tasks=uncached_tasks)


def main():
    """Entrypoint for Launching sagemaker jobs using CLI"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment_name', type=str, required=True, help="Name of the experiment")
    parser.add_argument('--context_name', type=str, default="D244_F3_C1530_30", help="Name of the context")
    parser.add_argument('--datasets', nargs='+', type=str, required=True, help="List of datasets to evaluate")
    parser.add_argument('--folds', nargs='+', type=int, required=True, help="List of folds to evaluate")
    parser.add_argument('--methods_file', type=str, required=True, help="Path to the YAML file containing methods")
    parser.add_argument('--max_concurrent_jobs', type=int, default=50,
                        help="Maximum number of concurrent jobs, based on account limit")
    parser.add_argument('--docker_image_uri', type=str, required=True, help="Docker image URI or alias in constants.py")
    parser.add_argument('--instance_type', type=str, default="ml.m6i.4xlarge", help="SageMaker instance type")
    parser.add_argument('--sagemaker_role', type=str, required=True, help="AWS IAM role ARN for SageMaker")
    parser.add_argument('--s3_bucket', type=str, required=True, help="S3 bucket for the experiment")
    parser.add_argument('--add_timestamp', action='store_true', help="Add timestamp to the experiment name")
    parser.add_argument('--no-wait', dest='wait', action='store_false', help="Skip waiting for jobs to complete")
    parser.set_defaults(wait=True)
    parser.add_argument('--batch_size', type=int, default=16, help="Batch size for tasks")
    parser.add_argument('--verbose', action='store_true', help="Enable verbose logging")
    parser.add_argument('--s3_dataset_cache', type=str, required=False, default=None,
                        help="S3 URI for OpenML dataset cache (format: s3://bucket/prefix)")
    
    args = parser.parse_args()

    launch_jobs(
        experiment_name=args.experiment_name,
        context_name=args.context_name,
        datasets=args.datasets,
        folds=args.folds,
        methods_file=args.methods_file,
        max_concurrent_jobs=args.max_concurrent_jobs,
        docker_image_uri=args.docker_image_uri,
        instance_type=args.instance_type,
        sagemaker_role=args.sagemaker_role,
        s3_bucket=args.s3_bucket,
        wait=args.wait,
        batch_size=args.batch_size,
        s3_dataset_cache=args.s3_dataset_cache,
    )


if __name__ == "__main__":
    main()
