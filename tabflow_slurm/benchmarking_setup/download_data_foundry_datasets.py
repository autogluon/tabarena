from __future__ import annotations

from pathlib import Path

from tabflow_slurm.benchmarking_setup.data_foundry_integration.data_foundry_task_creator import (
    DataFoundryAdapter,
)

DEFAULT_DATA_FOUNDRY_CACHE = Path(__file__).parent / ".data_foundry_cache"

EXAMPLE_DATA_FOUNDRY_TASKS = [
    # Grouped tiny data
    "musk/019cb408-670c-7088-bf5e-eb09cb01e9b2",
    # Temporal data
    "mercedes_benz_greener_manufacturing/019c0e8e-8749-7ff7-9c06-632c3ca2aa05",
    # IID Tabular Text Data
    "wine_world_cost/019c32f6-9391-7812-b543-66fbb299dc51",
]


def download_data_foundry_datasets(
    *, benchmark_suite_name: str, data_foundry_artifacts: list[str]
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

    Returns:
    -------
    pd.DataFrame
        A DataFrame containing metadata about the created TabArena UserTasks.
    """
    print("Preprocessing data foundry datasets for TabArena...")
    task_metadata = DataFoundryAdapter(
        data_foundry_artifacts=data_foundry_artifacts,
        path_to_data_foundry_cache=DEFAULT_DATA_FOUNDRY_CACHE,
    ).to_tabarena_user_tasks(show_sample=True)

    path_to_metadata = (
        DEFAULT_DATA_FOUNDRY_CACHE / f"{benchmark_suite_name}_tasks_metadata.csv"
    )
    print(f"Saving metadata to {path_to_metadata}")
    task_metadata.to_csv(path_to_metadata, index=False)


def get_metadata_for_benchmark_suite(benchmark_suite_name: str) -> Path:
    """Get the path to the metadata CSV file for a given benchmark suite.

    Parameters
    ----------
    benchmark_suite_name : str
        The name of the benchmark suite for which to retrieve the metadata.

    Returns:
    -------
    Path
        The path to the metadata CSV file for the specified benchmark suite.
    """
    path_to_metadata = (
        DEFAULT_DATA_FOUNDRY_CACHE / f"{benchmark_suite_name}_tasks_metadata.csv"
    )
    if not path_to_metadata.exists():
        raise FileNotFoundError(
            f"Metadata file {path_to_metadata} does not exist. "
            "Please run download_data_foundry_datasets first."
        )
    return path_to_metadata


if __name__ == "__main__":
    download_data_foundry_datasets(
        benchmark_suite_name="example_benchmark_suite",
        data_foundry_artifacts=EXAMPLE_DATA_FOUNDRY_TASKS,
    )
