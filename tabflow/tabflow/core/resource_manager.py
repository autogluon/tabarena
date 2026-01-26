import time
from tabflow.utils.logging_utils import save_training_job_logs


class TrainingJobResourceManager:
    """Class to manage SageMaker training job resources and monitors job status."""

    def __init__(self, sagemaker_client, max_concurrent_jobs):
        """
        Initialize the resource manager along with client and concurrency limit.
        """
        self.sagemaker_client = sagemaker_client
        self.max_concurrent_jobs = max_concurrent_jobs
        self.job_names = []
        self.job_cache_paths = {} # Job Name -> Training Log
        self.job_statuses = {'InProgress': 0, 'Completed': 0, 'Failed': 0, 'Stopped': 0}
        self.jobs_logged = 0
        self.total_jobs = 0
        self.launched_jobs = 0
        self.time_start = time.time()


    def add_job(self, job_name, cache_path):
        """
        Add a new job to be tracked by the manager.
        
        Args:
            job_name: Name of the SageMaker training job
            cache_path: Path where the job results will be cached
        """
        self.job_names.append(job_name)
        self.job_cache_paths[job_name] = cache_path
        self.launched_jobs += 1
        self.job_statuses['InProgress'] += 1


    def remove_completed_jobs(self, s3_client, s3_bucket):
        """
        Remove completed jobs from the tracking list and update status counters.
        
        Args:
            s3_client: Boto3 S3 client
            s3_bucket: S3 bucket name for storing logs
            
        Returns:
            Number of completed jobs removed
        """
        
        completed_jobs = []
        for job_name in self.job_names:
            response = self.sagemaker_client.describe_training_job(TrainingJobName=job_name)    #FIXME: Possible Throttling here if Queue is too big
            job_status = response['TrainingJobStatus']

            if job_status in ['Completed', 'Failed', 'Stopped']:

                self.job_statuses['InProgress'] -= 1
                if job_status == 'Completed':
                    self.job_statuses['Completed'] += 1
                elif job_status == 'Failed':
                    self.job_statuses['Failed'] += 1
                elif job_status == 'Stopped':
                    self.job_statuses['Stopped'] += 1

                # TODO: The below logic is slow, so it is disabled for now
                if False:
                    save_training_job_logs(
                        self.sagemaker_client,
                        s3_client,
                        job_name,
                        s3_bucket,
                        self.job_cache_paths[job_name]
                    )
                self.jobs_logged += 1
                completed_jobs.append(job_name)

        for job_name in completed_jobs:
            self.job_names.remove(job_name)
        
        return len(completed_jobs)


    def wait_for_available_slot(self, s3_client, s3_bucket, poll_interval=10):
        """
        Wait until a job slot becomes available.
        NOTE: Calling remove_completed_jobs() AFTER if len(self.job_names) < self.max_concurrent_jobs, otheriwse it leads to API throttling
        as it will keep on calling describe_training_job on the added job list which keeps expanding.

        Args:
            s3_client: Boto3 S3 client
            s3_bucket: S3 bucket name for storing logs
            poll_interval: Time in seconds between checks for available slots
        """
        while True:
            if len(self.job_names) < self.max_concurrent_jobs:
                self._print_status()
                print(f"Slot available...")
                return
            
            removed_jobs = self.remove_completed_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
            self._print_status()

            if removed_jobs == 0:
                print(f"Waiting for slot... Currently running {len(self.job_names)} out of {self.max_concurrent_jobs} concurrent jobs. ")
                time.sleep(poll_interval)


    def _print_status(self):
        """Print the current status of all jobs."""
        time_cur = time.time()
        time_elapsed = time_cur - self.time_start
        print("\r", end="")
        print(f"\nProcessed Jobs: {self.launched_jobs}/{self.total_jobs} | "
              f"[In Progress: {self.job_statuses['InProgress']}, "
              f"Completed: {self.job_statuses['Completed']}, "
              f"Failed: {self.job_statuses['Failed']}, "
              f"Stopped: {self.job_statuses['Stopped']}] | "
              f"Jobs Logged: {self.jobs_logged} | "
              f"Time Elapsed: {time_elapsed:7.1f}s | "
              f"Concurrency: {len(self.job_names)}/{self.max_concurrent_jobs}", end="")
        print()


    def wait_for_all_jobs(self, s3_client, s3_bucket, poll_interval=10, stop_wait_on_fail=False):
        """
        Wait for all jobs to complete.
        
        Args:
            s3_client: Boto3 S3 client
            s3_bucket: S3 bucket name for storing logs
            poll_interval: Time in seconds between status checks
        """
        # Wait for a non-zero value
        while self.job_names:
            removed_jobs = self.remove_completed_jobs(s3_client=s3_client, s3_bucket=s3_bucket)
            self._print_status()
            if stop_wait_on_fail and self.job_statuses['Failed'] > 0:
                # TODO: Consider implementing shutoff logic for the non-failed nodes
                return
            if removed_jobs == 0:
                print(f"Waiting for {len(self.job_names)} jobs to complete...")
                time.sleep(poll_interval)
