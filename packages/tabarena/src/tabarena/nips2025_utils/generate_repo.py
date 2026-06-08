from __future__ import annotations

from typing import TYPE_CHECKING

from tabarena.benchmark.result import BaselineResult, ConfigResult, ExperimentResults
from tabarena.benchmark.task.metadata.collection import TaskMetadataCollection
from tabarena.utils.pickle_utils import fetch_all_pickles

from .load_artifacts import load_all_artifacts

if TYPE_CHECKING:
    from pathlib import Path

    import pandas as pd

    from tabarena.repository import EvaluationRepository


def generate_repo(
    experiment_path: str, task_metadata: pd.DataFrame | TaskMetadataCollection, name_suffix: str | None = None
) -> EvaluationRepository:
    file_paths = fetch_all_pickles(dir_path=experiment_path)
    file_paths = sorted([str(f) for f in file_paths])
    print(len(file_paths))

    return generate_repo_from_paths(result_paths=file_paths, task_metadata=task_metadata, name_suffix=name_suffix)


def generate_repo_from_paths(
    result_paths: list[str | Path],
    task_metadata: pd.DataFrame | TaskMetadataCollection,
    engine: str = "ray",
    name_suffix: str | None = None,
    as_holdout: bool = False,
) -> EvaluationRepository:
    results_lst = load_all_artifacts(file_paths=result_paths, engine=engine, convert_to_holdout=as_holdout)
    return generate_repo_from_results_lst(
        results_lst=results_lst,
        task_metadata=task_metadata,
        name_suffix=name_suffix,
    )


def generate_repo_from_results_lst(
    results_lst: list[BaselineResult],
    task_metadata: pd.DataFrame | TaskMetadataCollection,
    name_suffix: str | None = None,
) -> EvaluationRepository:
    results_lst = [r for r in results_lst if r is not None]

    # A native TaskMetadataCollection supplies the tid filter via its dataset->tid map; the
    # legacy DataFrame the tabrepo core (ExperimentResults) requires is derived once here, at
    # this single boundary, via to_legacy_df(). A raw DataFrame is used as-is (legacy callers).
    if isinstance(task_metadata, TaskMetadataCollection):
        tids = set(task_metadata.dataset_to_tid().values())
        task_metadata = task_metadata.to_legacy_df()
    else:
        tids = set(task_metadata["tid"].unique())
    assert all(not isinstance(tid, str) for tid in tids), f"Expected all tids to be numbers, but got str: {tids}"
    results_lst = [r for r in results_lst if r.result["task_metadata"]["tid"] in tids]

    if name_suffix is not None:
        for r in results_lst:
            r.update_name(name_suffix=name_suffix)
            if isinstance(r, ConfigResult):
                r.update_model_type(name_suffix=name_suffix)

    if len(results_lst) == 0:
        raise ValueError(
            "No results found after filtering by task metadata tids. "
            "Please check that the tids in the results match those in the task metadata. "
            "Moreover, check that the path you are parsing only contain results for the allowed tids!",
        )

    exp_results = ExperimentResults(task_metadata=task_metadata)

    repo: EvaluationRepository = exp_results.repo_from_results(results_lst=results_lst)
    return repo
