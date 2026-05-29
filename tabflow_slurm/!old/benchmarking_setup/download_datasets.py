"""Example Usage.

> python download_datasets.py --directory /path/to/.openml-cache --action download
"""

from __future__ import annotations

import argparse
from pathlib import Path

import openml
import tqdm

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download benchmark datasets")
    parser.add_argument(
        "--directory",
        type=Path,
        default=Path("../openml-cache"),
        help="Directory to save the datasets",
    )
    parser.add_argument(
        "--task_suite",
        type=str,
        default="tabarena-v0.1",
        help="Name of the OpenML suite for the tasks to use, e.g. tabarena-v0.1",
    )
    parser.add_argument("--action", choices=["download", "list"])
    args = parser.parse_args()

    try:
        openml.config.set_cache_directory(str(args.directory.resolve().absolute()))
    except Exception:
        openml.config._root_cache_directory = str(args.directory.resolve().absolute())

    tasks = openml.study.get_suite(args.task_suite).tasks

    match args.action:
        case "download":
            for task in tqdm.tqdm(tasks):
                openml.tasks.get_task(task, download_data=True, download_qualities=True, download_splits=True)
        case "list":
            for task in tasks:
                print(task)
