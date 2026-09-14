from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import TYPE_CHECKING

import openml
from openml import OpenMLSupervisedTask
from openml.exceptions import OpenMLServerException

if TYPE_CHECKING:
    import pandas as pd
    from openml import OpenMLDataset

logger = logging.getLogger(__name__)


def get_task(task_id: int) -> OpenMLSupervisedTask:
    task = openml.tasks.get_task(
        task_id,
        download_splits=False,
        download_data=True,
        download_qualities=True,
        download_features_meta_data=True,
    )
    if isinstance(task, OpenMLSupervisedTask):
        return task
    raise AssertionError(f"Invalid task type: {type(task)}")


def get_ag_problem_type(task: OpenMLSupervisedTask) -> str:
    if task.task_type_id.name == "SUPERVISED_CLASSIFICATION":
        problem_type = "multiclass" if len(task.class_labels) > 2 else "binary"
    elif task.task_type_id.name == "SUPERVISED_REGRESSION":
        problem_type = "regression"
    else:
        raise AssertionError(f"Unsupported task type: {task.task_type_id.name}")
    return problem_type


def get_task_with_retry(task_id: int, max_delay_exp: int = 8) -> OpenMLSupervisedTask:
    delay_exp = 0
    while True:
        try:
            # print(f'Getting task {task_id}')
            return get_task(task_id=task_id)
            # print(f'Got task {task_id}')
        except OpenMLServerException as e:
            delay = 2**delay_exp
            delay_exp += 1
            if delay_exp > max_delay_exp:
                raise ValueError("Unable to get task after 10 retries") from e
            print(e)
            print(f"Retry in {delay}s...")
            time.sleep(delay)
            continue


def use_cached_pickle(dataset: OpenMLDataset) -> None:
    """Point ``dataset`` at the pickle cache openml wrote for it on an earlier load.

    openml registers ``dataset_<id>.pkl.py3`` on the dataset object only when the dataset
    was constructed from an ARFF file. With a parquet-backed cache (the default since
    openml-python 0.14) the pickle path stays ``None``, so every ``get_data`` call re-parses
    the parquet and rewrites the pickle instead of loading it. Registering the existing
    pickle makes ``get_data`` load it directly; a missing or unreadable pickle falls back
    to openml's own parquet path.
    """
    if dataset.cache_format != "pickle" or dataset.data_pickle_file is not None:
        return
    source = dataset.parquet_file or dataset.data_file
    if source is None:
        return
    pickle_file = Path(source).with_suffix(".pkl.py3")
    if pickle_file.exists():
        dataset.data_pickle_file = str(pickle_file)


def get_task_data(task: OpenMLSupervisedTask) -> tuple[pd.DataFrame, pd.Series]:
    dataset = task.get_dataset(download_data=True)
    use_cached_pickle(dataset)
    X, y, _, _ = dataset.get_data(task.target_name)
    return X, y
