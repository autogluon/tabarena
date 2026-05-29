# Benchmarking Setup

This directory contains the code to run before benchmarking to set up the environment and datasets correctly.

* `download_all_foundation_models.py`: code to download all foundation models we benchmark to the local system. This
  should be used within the environment used to run the benchmarking code and avoids race conditions when multiple
  processes would try to download the same model.
* `download_datasets.py`: code to pre-download and cache all datasets we benchmark to the local system. To avoid race
  conditions when multiple processes would try to download the same dataset, this should be run once before starting
  parallel benchmarking. Moreover, we recommend to downland this to a share persistent drive that all jobs can access (
  e.g., a SLURM workspace)