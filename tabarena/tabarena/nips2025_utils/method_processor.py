from __future__ import annotations

import io
from collections import Counter
from pathlib import Path

import pandas as pd

from tabarena.benchmark.result import BaselineResult, ConfigResult, AGBagResult
from tabarena.nips2025_utils.load_artifacts import load_all_artifacts
from tabarena.utils.pickle_utils import fetch_all_pickles


def generate_task_metadata(tids: list[int]) -> pd.DataFrame:
    """
    Retrieve metadata for a list of OpenML task IDs and return as a clean DataFrame.

    This function queries OpenML to obtain task metadata for the specified list of task IDs (`tids`),
    verifies that all requested tasks exist in OpenML, and returns a pandas DataFrame with the metadata.
    It also ensures compatibility with Parquet format by writing and reading the DataFrame through an
    in-memory CSV buffer to convert any complex column types into simple types.

    Parameters
    ----------
    tids : list of int
        List of OpenML task IDs to retrieve metadata for.

    Returns
    -------
    pd.DataFrame
        DataFrame containing metadata for the requested task IDs.

    Raises
    ------
    AssertionError
        If one or more of the requested task IDs are not found in OpenML's task list.
    """
    import openml

    tasks = openml.tasks.list_tasks(
        output_format="dataframe"
    )

    tasks_filtered = tasks[tasks["tid"].isin(tids)].reset_index(drop=True)
    tids_filtered = list(tasks_filtered["tid"])
    tids_missing = [tid for tid in tids if tid not in tids_filtered]
    if tids_missing:
        raise AssertionError(
            f"Missing {len(tids_missing)}/{len(tids)} tids in OpenML, for some reason `openml.tasks.list_tasks` "
            f"does not contain these tids:\n\t{tids_missing}"
        )

    # Convert to simple types to be able to save in parquet without issue
    csv_buffer = io.StringIO()
    tasks_filtered.to_csv(csv_buffer, index=False)

    # Move to the beginning of the buffer
    csv_buffer.seek(0)

    # Load DataFrame back from in-memory CSV
    tasks_filtered = pd.read_csv(csv_buffer)

    return tasks_filtered


def get_info_from_result(result: BaselineResult) -> dict:
    cur_task_metadata = result.task_metadata
    cur_result = dict()
    cur_result["framework"] = result.framework
    cur_result["metric"] = result.result["metric"]
    cur_result["problem_type"] = result.problem_type
    cur_result.update(cur_task_metadata)
    is_bag = False

    ag_key = None
    model_type = None
    name_prefix = None
    num_gpus = 0
    method_type = "baseline"
    if isinstance(result, ConfigResult):
        hyperparameters = result.hyperparameters
        model_cls = hyperparameters["model_cls"]
        model_type = hyperparameters["model_type"]
        ag_key = hyperparameters["ag_key"]
        name_prefix = hyperparameters["name_prefix"]
        num_gpus = result.result["method_metadata"].get("num_gpus", 0)
        method_type = "config"

        if isinstance(result, AGBagResult):
            if result.num_children > 1:
                is_bag = True

    cur_result["is_bag"] = is_bag
    cur_result["ag_key"] = ag_key
    cur_result["model_type"] = model_type
    cur_result["method_type"] = method_type
    cur_result["name_prefix"] = name_prefix
    cur_result["num_gpus"] = num_gpus

    return cur_result


def load_raw(
    path_raw: str | Path | list[str | Path] = None,
    engine: str = "ray",
    as_holdout: bool = False,
) -> list[BaselineResult]:
    """
    Loads the raw results artifacts from all `results.pkl` files in the `path_raw` directory

    Parameters
    ----------
    path_raw
    engine
    as_holdout

    Returns
    -------

    """

    suffix = "results.pkl"
    file_paths_method = fetch_all_pickles(dir_path=path_raw, suffix=suffix)
    if len(file_paths_method) == 0:
        # Look at every file to provide debugging info
        all_files = [p for p in Path(path_raw).rglob("*") if p.is_file()]

        # Root name = filename with extension
        roots = [p.name for p in all_files]

        counter = Counter(roots).most_common(10)
        common_str = ", ".join(f"{name} ({count})" for name, count in counter) or "None"

        raise AssertionError(
            f"No valid {suffix!r} files found under path: {path_raw!r}!\n"
            f"Most common file root names:\n{common_str}"
        )
    results_lst = load_all_artifacts(file_paths=file_paths_method, engine=engine, convert_to_holdout=as_holdout)
    return results_lst
