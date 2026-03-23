from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from data_foundry.curation_container import CuratedContainer
from data_foundry.schema import ProblemTypeClassification
from loguru import logger
from tabarena.benchmark.task import UserTask
from tqdm import tqdm

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
    """Dataframe version of user_tasks.TabArenaTaskMetadata.
    """

    def __init__(
        self,
        *,
        data_foundry_artifacts: list[str],
        path_to_data_foundry_cache: Path,
        evaluation_metrics: dict[str, list[str]] | None = None,
    ) -> None:
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
        task_metadata = None

        for df_artifact_name_id in tqdm(
            self.data_foundry_artifacts, desc="Caching tasks and saving metadata..."
        ):
            task = convert_data_foundry_task_to_user_task(
                path_to_local_task=self.path_to_data_foundry_cache
                / df_artifact_name_id,
                evaluation_metrics=self.evaluation_metrics,
            )
            oml_task = task.load_local_openml_task()
            tabarena_task_name = task.tabarena_task_name
            task_id_str = task.task_id_str
            del task

            new_task_metadata = oml_task.compute_metadata().to_dataframe()
            new_task_metadata["tabarena_task_name"] = tabarena_task_name
            new_task_metadata["task_id_str"] = task_id_str

            if task_metadata is None:
                task_metadata = new_task_metadata
            else:
                task_metadata = pd.concat(
                    [task_metadata, new_task_metadata], ignore_index=True
                )

        self._user_tasks_metadata = task_metadata

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
            logger.info(
                f"Objective metric {eval_metric} not in allowed for problem type {problem_type}. "
                f"Falling back to {fallback_metric}."
            )
            eval_metric = fallback_metric

    # Create UserTask
    user_task = UserTask(
        task_name=task_container.unique_name,
    )
    oml_task = user_task.create_local_openml_task(
        dataset=task_container.dataset,
        target_feature=task_container.task_metadata.target_column_name,
        problem_type=problem_type,
        splits=task_container.experiment_metadata.splits,
        eval_metric=eval_metric,
        stratify_on=task_container.task_metadata.stratify_on,
        group_on=task_container.task_metadata.group_on,
        time_on=task_container.task_metadata.time_on,
        group_time_on=task_container.task_metadata.group_time_on,
        dataset_name=task_container.dataset_metadata.unique_name,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task


if __name__ == "__main__":
    task_metadata = DataFoundryAdapter(
        data_foundry_artifacts=[
            # Grouped tiny data
            "musk/019cb408-670c-7088-bf5e-eb09cb01e9b2",
            # Temporal data
            "mercedes_benz_greener_manufacturing/019c0e8e-8749-7ff7-9c06-632c3ca2aa05",
            # IID Tabular Text Data
            "wine_world_cost/019c32f6-9391-7812-b543-66fbb299dc51",
        ],
        path_to_data_foundry_cache=Path(__file__).parent / ".data_foundry_cache",
    ).to_tabarena_user_tasks()

    # Print Example
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)
    print(task_metadata)
