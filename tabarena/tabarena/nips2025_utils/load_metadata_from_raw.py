from __future__ import annotations

from typing import Literal
from pathlib import Path

from tabarena.benchmark.result import BaselineResult
from tabarena.nips2025_utils.artifacts.method_metadata import MethodMetadata
from tabarena.utils.parallel_for import parallel_for


def load_raw_to_get_metadata(path: str) -> tuple[MethodMetadata, dict]:
    """
    Load a raw result pickle file, align it into a `BaselineResult`,
    and derive the associated `MethodMetadata` and task metadata.

    Parameters
    ----------
    path : str
        Path to the pickle file containing either a raw result
        dictionary or a serialized `BaselineResult`.

    Returns
    -------
    tuple[MethodMetadata, dict]
        A tuple containing:
        - `MethodMetadata`: Metadata object constructed from the result.
        - dict: The task metadata dictionary extracted from the result.

    Notes
    -----
    This function ensures that raw results are normalized into
    `BaselineResult` objects before metadata extraction. It should
    be used whenever raw pickled results need to be converted into
    higher-level metadata representations.
    """
    result = BaselineResult.from_pickle(path=path)
    method_metadata = MethodMetadata.from_raw([result])
    task_metadata = result.task_metadata
    return method_metadata, task_metadata


def load_from_raw_all_metadata(
    file_paths: list[str | Path],
    engine: Literal["sequential", "ray", "joblib"] = "sequential",
    progress_bar: bool = True,
) -> list[tuple[MethodMetadata, dict]]:
    """
    Batch-load multiple raw pickle results and extract their
    `MethodMetadata` and task metadata in parallel.

    Parameters
    ----------
    file_paths : list[str or pathlib.Path]
        List of paths to pickle files containing raw results.
    engine : Literal["sequential", "ray", "joblib"], default="sequential"
        Parallelization engine to use for processing.
    progress_bar : bool, default=True
        Whether to display a progress bar while processing.

    Returns
    -------
    list[tuple[MethodMetadata, dict]]
        A list of tuples, one per input file, each containing:
        - `MethodMetadata`: Metadata object extracted from the result.
        - dict: Task metadata dictionary.

    Notes
    -----
    This function leverages `parallel_for` to parallelize calls to
    `load_raw_to_get_metadata`. The parallelization backend can be
    configured via the `engine` parameter.
    """
    file_paths_lst = []
    for file_path in file_paths:
        file_paths_lst.append(
            {
                "path": str(file_path),
            }
        )

    results_lst: list[tuple[MethodMetadata, dict]] = parallel_for(
        f=load_raw_to_get_metadata,
        inputs=file_paths_lst,
        engine=engine,
        progress_bar=progress_bar,
    )
    return results_lst
