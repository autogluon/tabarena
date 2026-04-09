from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pandas as pd
from data_foundry.curation_container import CuratedContainer
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

    def to_tabarena_user_tasks(self, *, show_sample: bool = False) -> pd.DataFrame:
        """Converts the specified Data Foundry artifacts into TabArena UserTasks.

        Parameters
        ----------
        show_sample: bool
            If True, shows a sample of the created UserTasks metadata.
            Useful for debugging.

        Returns a DataFrame containing metadata about the created UserTasks.
        """
        task_metadata = None

        for data_foundry_uri in tqdm(self.data_foundry_artifacts, desc="Caching tasks and saving metadata..."):
            task = convert_data_foundry_task_to_user_task(
                path_to_local_task=self.path_to_data_foundry_cache / data_foundry_uri,
                evaluation_metrics=self.evaluation_metrics,
            )
            oml_task = task.load_local_openml_task()
            tabarena_task_name = task.tabarena_task_name
            task_id_str = task.task_id_str
            del task

            new_task_metadata = oml_task.compute_metadata(
                tabarena_task_name=tabarena_task_name,
                task_id_str=task_id_str,
            ).to_dataframe()
            new_task_metadata["data_foundry_uri"] = data_foundry_uri

            if task_metadata is None:
                task_metadata = new_task_metadata
            else:
                task_metadata = pd.concat([task_metadata, new_task_metadata], ignore_index=True)

        self._user_tasks_metadata = task_metadata

        if show_sample:
            with pd.option_context(
                "display.max_rows",
                None,
                "display.max_columns",
                None,
                "display.width",
                None,
                "display.max_colwidth",
                None,
            ):
                print(task_metadata.sample(min(5, len(task_metadata)), random_state=42))

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
    y: pd.Series = task_container.dataset[task_container.task_metadata.target_column_name]
    if task_container.task_metadata.problem_type == "regression":
        problem_type = "regression"
        # Assert y is pd.numeric
        if not pd.api.types.is_numeric_dtype(y):
            raise ValueError(
                f"Target column {task_container.task_metadata.target_column_name} is not numeric for "
                f"regression problem. ({task_container.dataset_metadata.unique_name})"
            )
    elif task_container.task_metadata.problem_type == "binary_classification":
        problem_type = "classification"
        # Assert y is pd.categorical with 2 classes
        if not isinstance(y.dtype, pd.CategoricalDtype):
            raise ValueError(
                f"Target column {task_container.task_metadata.target_column_name} is not categorical "
                f"for classification problem. ({task_container.dataset_metadata.unique_name})"
            )
        if y.nunique() != 2:
            raise ValueError(
                f"Target column {task_container.task_metadata.target_column_name} has {y.nunique()} classes, "
                f"but expected 2 for binary classification problem. ({task_container.dataset_metadata.unique_name})"
            )
    elif task_container.task_metadata.problem_type == "multiclass_classification":
        problem_type = "classification"
        if not isinstance(y.dtype, pd.CategoricalDtype):
            raise ValueError(
                f"Target column {task_container.task_metadata.target_column_name} is not categorical for classification problem."
            )
        if y.nunique() < 3:
            raise ValueError(
                f"Target column {task_container.task_metadata.target_column_name} has {y.nunique()} classes, "
                "but expected at least 3 for multiclass classification problem."
            )
    else:
        raise ValueError(f"Unknown problem type {task_container.task_metadata.problem_type}")

    # Resolve eval metric
    eval_metric = task_container.task_metadata.objective_metric_name
    if evaluation_metrics is not None:
        allowed_eval_metrics = evaluation_metrics[task_container.task_metadata.problem_type]
        fallback_metric = allowed_eval_metrics[0]
        if eval_metric not in allowed_eval_metrics:
            logger.info(
                f"\nObjective metric {eval_metric} not allowed for problem type {problem_type}. "
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
        group_labels=task_container.task_metadata.group_labels,
        split_time_horizon=task_container.experiment_metadata.time_horizon,
        split_time_horizon_unit=task_container.experiment_metadata.time_horizon_unit,
    )
    user_task.save_local_openml_task(oml_task)
    return user_task


def download_data_foundry_datasets(
    *,
    benchmark_suite_name: str,
    data_foundry_artifacts: list[str],
    data_foundry_cache: Path,
    openml_cache: str | Path | None = None,
):
    """Prepare data foundry artifacts for TabArena usage.

    1) Converts Data foundry artifacts into TabArena UserTasks and saves them on disk
    2) Collects metadata about the created UserTasks
    3) Saves the metadata in a standardized location for later use.

    Parameters
    ----------
    benchmark_suite_name:
        Name of the benchmark suite for which the datasets are being downloaded.
        This is used to store the metadata of the datasets in a standardized location.
    data_foundry_artifacts : list[str]
        A list of Data Foundry artifact identifiers to be downloaded and converted into
        TabArena UserTasks.
        Each identifier should be in the format "dataset_name/artifact_uuid".
    data_foundry_cache : Path
        The path to the cache directory where to store data foundry data.
    openml_cache: str | None
        If not None, sets the OpenML cache directory to the specified path for
        downloading and caching TabArenaOpenML tasks.

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing metadata about the created TabArena UserTasks.
    """
    if openml_cache is not None:
        import openml

        print(f"Setting OpenML cache directory to: {openml_cache}")
        openml.config.set_root_cache_directory(root_cache_directory=openml_cache)

    print("Preprocessing data foundry datasets for TabArena...")
    task_metadata = DataFoundryAdapter(
        data_foundry_artifacts=data_foundry_artifacts,
        path_to_data_foundry_cache=data_foundry_cache,
    ).to_tabarena_user_tasks(show_sample=True)

    path_to_metadata = data_foundry_cache / f"{benchmark_suite_name}_tasks_metadata.csv"
    print(f"Saving metadata to {path_to_metadata}")
    task_metadata.to_csv(path_to_metadata, index=False)


def get_metadata_for_benchmark_suite(benchmark_suite_name: str, data_foundry_cache: Path) -> Path:
    """Get the path to the metadata CSV file for a given benchmark suite.

    Parameters
    ----------
    benchmark_suite_name : str
        The name of the benchmark suite for which to retrieve the metadata.
    data_foundry_cache : Path
        The path to the cache directory from where to load data foundry data.

    Returns:
    -------
    Path
        The path to the metadata CSV file for the specified benchmark suite.
    """
    path_to_metadata = data_foundry_cache / f"{benchmark_suite_name}_tasks_metadata.csv"
    if not path_to_metadata.exists():
        raise FileNotFoundError(
            f"Metadata file {path_to_metadata} does not exist. Please run download_data_foundry_datasets first."
        )
    return path_to_metadata


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
        path_to_data_foundry_cache=Path(__file__).parent.parent / ".data_foundry_cache",
    ).to_tabarena_user_tasks(show_sample=True)
