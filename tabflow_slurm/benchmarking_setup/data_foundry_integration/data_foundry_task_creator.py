from __future__ import annotations

from pathlib import Path

import pandas as pd
from tabarena.benchmark.task import UserTask
from data_foundry.curation_container import CuratedContainer
from data_foundry.schema import ProblemTypeClassification
from tqdm import tqdm
from copy import deepcopy
from tabarena.benchmark.task.user_task import get_tabarena_metadata_from_openml_task
from loguru import logger


DEFAULT_EVAL_METRICS = {
    "binary_classification": ["roc_auc"],
    "multiclass_classification": ["log_loss"],
    "regression": ["root_mean_squared_error"],
}

# TODO:
#  - support downloading/caching from buckets
class DataFoundryAdapter:
    """Class to process and convert Data Foundry artifacts for usage in TabArena."""


    _user_tasks_metadata: pd.DataFrame | None
    """The metadata for the UserTasks created from the Data Foundry artifacts, stored
    as a DataFrame with columns:
        "name", - > UserTask.tabarena_task_name
        "tabarena_num_repeats",
        "num_folds",
        "task_id", -> UserTask.task_id_str
        "num_instances",
        "num_features",
        "num_classes",
        "problem_type"
    """

    def __init__(self, *, data_foundry_artifacts: list[str], path_to_data_foundry_cache: Path, evaluation_metrics: dict[str, list[str]] | None = None) -> None:
        """Initializes the DataFoundryAdapter.

        Parameters
        ----------
        data_foundry_artifacts : list[str]
            A list of Data Foundry artifact identifiers to be processed.
        path_to_data_foundry_cache : Path
            The local path where Data Foundry artifacts are cached for processing.
        evaluation_metrics: dict[str, list[str]] | None
            A dictionary mapping problem types to lists of allowed evaluation metrics.
            If a curation container's objective metric is not in the allowed list for
                its problem type, we fall back to the first metric in the allowed list.
            If None, we default to a set of metrics as defined in DEFAULT_EVAL_METRICS.
        """
        self.data_foundry_artifacts = data_foundry_artifacts
        self.path_to_data_foundry_cache = path_to_data_foundry_cache

        if evaluation_metrics is None:
            evaluation_metrics = deepcopy(DEFAULT_EVAL_METRICS)
        self.evaluation_metrics = evaluation_metrics
        self._user_tasks_metadata = None

    def to_tabarena_user_tasks(self) -> pd.DataFrame:
        """Converts the specified Data Foundry artifacts into TabArena UserTasks.

        Returns a DataFrame containing metadata about the created UserTasks.
        """
        task_metadata = []

        for task in tqdm(self.data_foundry_artifacts, desc="Caching tasks and saving metadata..."):
            task = convert_data_foundry_task_to_user_task(
                path_to_local_task=self.path_to_data_foundry_cache / task,
                evaluation_metrics=self.evaluation_metrics,
            )
            oml_task = task.load_local_openml_task()


            get_tabarena_metadata_from_openml_task(
                oml_task=oml_task,
            )


            # Get training sizes per split from the OpenML task's dataset
            oml_dataset, *_ = oml_task.get_dataset().get_data()



            # TODO: update in the future for non-iid tasks that do not do normal outer CV.
            oml_dataset, *_ = oml_task.get_dataset().get_data()
            num_instances = oml_dataset.shape[0]
            num_features = oml_dataset.shape[1] - 1  # exclude target
            num_classes = (
                -1
                if oml_task.task_type_id.value == 2
                else oml_dataset[oml_task.target_name].nunique()
            )
            if num_classes == -1:
                problem_type = "regression"
            elif num_classes == 2:
                problem_type = "binary"
            else:
                problem_type = "multiclass"
            del oml_dataset

            task_metadata.append(
                [
                    task.tabarena_task_name,
                    oml_task.split.repeats,
                    oml_task.split.folds,
                    task.task_id_str,
                    num_instances,
                    num_features,
                    num_classes,
                    problem_type,
                ]
            )

        self._user_tasks_metadata = pd.DataFrame(
            task_metadata,
            columns=[
                "name",
                "tabarena_num_repeats",
                "num_folds",
                "task_id",
                "num_instances",
                "num_features",
                "num_classes",
                "problem_type",
            ],
        )

        return self._user_tasks_metadata


def convert_data_foundry_task_to_user_task(
    *,
    path_to_local_task: Path,
    evaluation_metrics: dict[str, list[str]] | None = None,
) -> UserTask:
    """Converts a Data Foundry task, stored locally as a CuratedContainer, into a TabArena UserTask.

    Parameters
    ----------
    path_to_local_task : Path
        The local file path to the Data Foundry artifact (from a CuratedContainer).
    evaluation_metrics: list[str]
        A mapping from problem type to allowed evaluation metrics.
        If a curation container's objective metric is not in the allowed list,
        we fall back to the first metric in the allowed list for that problem type.
        If None, we just use the eval_metric in the curation container.
    """
    task_container = CuratedContainer.load(path_to_local_task)

    # Resolve task type
    if task_container.task_metadata.problem_type == "regression":
        problem_type = "regression"
    elif task_container.task_metadata.problem_type in ProblemTypeClassification:
        problem_type = "classification"
    else:
        raise ValueError(
            f"Unknown problem type {task_container.task_metadata.problem_type}"
        )

    # Resolve eval metric
    eval_metric = task_container.task_metadata.objective_metric_name
    if evaluation_metrics is None:
        allowed_eval_metrics = evaluation_metrics[problem_type]
        fallback_metric = allowed_eval_metrics[0]
        if eval_metric not in allowed_eval_metrics:
            logger.info(f"Objective metric {eval_metric} not in allowed for problem type {problem_type}. "
                        f"Falling back to {fallback_metric}.")
            eval_metric = fallback_metric

    # Create UserTask
    user_task = UserTask(
        task_name=task_container.dataset_metadata.unique_name,
    )
    oml_task = user_task.create_local_openml_task(
        dataset=task_container.dataset,
        target_feature=task_container.task_metadata.target_column_name,
        problem_type=problem_type,
        splits=task_container.experiment_metadata.splits,
        eval_metric=eval_metric,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task


def get_local_task_metadata() -> pd.DataFrame:




def create_tabarena_local_task_metadata_file() -> None:
    task_metadata = get_local_task_metadata()
    task_metadata.to_csv(get_path_to_tabarena_local_task_metadata_file(), index=False)


def get_tabarena_local_task_metadata() -> pd.DataFrame:
    return pd.read_csv(get_path_to_tabarena_local_task_metadata_file())


def get_path_to_tabarena_local_task_metadata_file() -> Path:
    return SAVE_PATH / SAVE_NAME


if __name__ == "__main__":
    DataFoundryAdapter(
        data_foundry_artifacts=[
            # Grouped tiny data
            "musk/019cb408-670c-7088-bf5e-eb09cb01e9b2",
            # Temporal data
            "mercedes_benz_greener_manufacturing/019c0e8e-8749-7ff7-9c06-632c3ca2aa05",
            # IID Tabular Text Data
            "wine_world_cost/019c32f6-9391-7812-b543-66fbb299dc51",
        ],
        path_to_data_foundry_cache=Path(__file__).parent / ".data_foundry_cache",
    )